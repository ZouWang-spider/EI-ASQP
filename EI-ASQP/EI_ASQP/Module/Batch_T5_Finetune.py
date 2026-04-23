import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchcrf import CRF
from torch.utils.data import DataLoader
from peft import get_peft_model, AdaLoraConfig, TaskType
from transformers import T5Tokenizer, T5ForConditionalGeneration

from Causal_ASQP.DataProcess.Dataprocess import process_dataset
from Causal_ASQP.BaseModel.Prompt import construct_prompt
from Causal_ASQP.BaseModel.Co_attention import CoAttention
from Causal_ASQP.BaseModel.Token_Process import Tokenizer_Label
from Causal_ASQP.BaseModel.CGNN_Element import extract_quad_positions, CGNN_Compute
from Causal_ASQP.BaseModel.CGNN import CausalGenerationModel
from Causal_ASQP.BaseModel.Quad_Token_Tag import get_quad_token_list, calculate_quad_token_loss

from Causal_ASQP.BaseModel.F1_compute import compute_prf1


# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.is_available())


# 选择模型（使用 't5-small', 't5-based','t5-large', 但需更多显存 't5-3b', 't5-11b'）
from peft import PeftConfig, PeftModel
from transformers import T5ForConditionalGeneration, AutoTokenizer

# 1. 加载配置
config = PeftConfig.from_pretrained("D:/Project/Causal_ASQP/ModelPath/LapTop1")

# 2. 加载基础模型
base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)

# 3. 正确封装 PEFT 模型
finetuned_model = PeftModel.from_pretrained(
    model=base_model,  # 这里传模型对象，而不是字符串路径
    model_id="D:/Project/Causal_ASQP/ModelPath/LapTop1",
    is_trainable=True  # 关键修复！
)

# # 4. 可选：切换 adapter（不一定需要调用这个，如果只有一个 adapter 默认就好）
# finetuned_model.set_active_adapter("default")  # 正确调用

# # 冻结 decoder 参数（第二阶段只训练 encoder）
# for param in finetuned_model.decoder.parameters():
#     param.requires_grad = False

# 5. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 6. 放到设备上
finetuned_model.to(device)

# 数据集处理和encoding
class QuadDataset():
    def __init__(self, file_path, tokenizer, max_input_len=512, max_target_len=128):
        self.dataset, aspect_opinion_label = process_dataset(file_path)  # 数据集处理
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.aspect_opinion_label = aspect_opinion_label  # 包含标签信息的部分

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence, tokens, reordered_quads, target_text = self.dataset[idx]
        multi_label = self.aspect_opinion_label[idx]

        target_tokens, element_positions = extract_quad_positions(target_text)


        # 构建输入文本（Prompt）
        input_text = construct_prompt(sentence)  # 假设已定义该函数来构造输入文本
        # print(sentence)


        # 使用T5 tokenizer对输入和目标进行编码
        input_encoding = self.tokenizer(input_text, padding="max_length", truncation=True,
                                        max_length=self.max_input_len, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, padding="max_length", truncation=True,
                                         max_length=self.max_target_len, return_tensors="pt")


        # 获取输入ID和目标ID
        input_ids = input_encoding["input_ids"].squeeze(0)  # 变成 (seq_len,)
        attention_mask = input_encoding["attention_mask"].squeeze(0)

        # 目标标签处理：忽略PAD token
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # 设置PAD为-100，便于忽略PAD token计算损失

        # 将tokens返回，用于后续处理
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 依据T5编码器的token来设置aspect_mask, opinion_mask, seq_labels
        aspect_mask, opinion_mask, seq_labels = Tokenizer_Label(tokens, reordered_quads)

        quad_token_list = get_quad_token_list(tokens, reordered_quads)

        return {
            'input_text': input_text,  # 真实目标文本字符串
            'tokens': tokens,  # 输入文本的token列表
            'input_ids': input_ids,  # 输入ID（句子对应的token IDs）
            'attention_mask': attention_mask,  # attention mask
            'labels': labels.squeeze(0),  # 目标标签
            'target_text': target_text,  # 目标文本字符串（用于可视化等）
            'element_positions': element_positions, #情感四元素在目标文本中的位置信息
            'multi_label': multi_label,  # 四分类标签（EA&EO, EA&IO, IA&EO, IA&IO）
            'aspect_mask': aspect_mask,  # 方面词mask
            'opinion_mask': opinion_mask,  # 观点词mask
            'seq_labels': seq_labels,  # 方面-观点联合标注
            'quad_token_list': quad_token_list #quad-token标签组合
        }


# 自定义 collate_fn
def collate_fn(batch):
    # batch 是一个 list，每个元素是 dataset 返回的字典
    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)  # [batch, seq_len]
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    labels = torch.stack([item['labels'] for item in batch], dim=0)

    # multi_label 可能是 list，需要转成 tensor
    multi_label = torch.tensor([item['multi_label'] for item in batch], dtype=torch.float32)

    # # aspect/opinion mask
    # aspect_mask = torch.stack([item['aspect_mask'] for item in batch], dim=0)
    # opinion_mask = torch.stack([item['opinion_mask'] for item in batch], dim=0)

    # tokens, target_text, element_positions, seq_labels, quad_token_list 可以保持 list
    tokens = [item['tokens'] for item in batch]
    target_text = [item['target_text'] for item in batch]
    # element_positions = [item['element_positions'] for item in batch]
    # seq_labels = [item['seq_labels'] for item in batch]
    # quad_token_list = [item['quad_token_list'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'multi_label': multi_label,
        # 'aspect_mask': aspect_mask,
        # 'opinion_mask': opinion_mask,
        'tokens': tokens,
        'target_text': target_text,
        # 'element_positions': element_positions,
        # 'seq_labels': seq_labels,
        # 'quad_token_list': quad_token_list
    }


# ---------- T5 模型的微调训练开始 ----------
file_path = r"D:\Project\Causal_ASQP\Dataset\Laptop\train.txt"
train_dataset = QuadDataset(file_path, tokenizer)

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=12,  # 你想要的 batch_size
    shuffle=True,
    collate_fn=collate_fn
)


# 初始化 Co-attention 模型
# co_attention_model = CoAttention(d_model=768).to(device)

BCE_criterion = nn.BCEWithLogitsLoss().to(device)

#多标签分类任务
# fc_layer = nn.Sequential(
#     nn.Linear(768, 256),
#     nn.ReLU(),
#     nn.Linear(256, 4)
# ).to(device)

#CRF序列标注
# crf = CRF(3, batch_first=True).to(device)
# linear = nn.Sequential(
#     nn.Linear(768, 256),  # 第一层
#     nn.ReLU(),            # 激活函数
#     nn.Linear(256, 3)     # 第二层
# ).to(device)

#因果生成神经网络初始化 CGNN
# CGNN_model = CausalGenerationModel(embed_dim=768, hidden_dim=512).to(device)


#CE 损失
CE_fn = torch.nn.CrossEntropyLoss().to(device)

# 训练循环
optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=5e-5)  # 选择合适的学习率
# # 需要优化的所有模型的参数
# optimizer = torch.optim.Adam(
#     list(finetuned_model.parameters()) +  # finetuned_model的参数
#     list(co_attention_model.parameters()) +  # CoAttention的参数
#     list(fc_layer.parameters()) +  # fc_layer的参数
#     list(linear.parameters()) +  # linear的参数
#     list(CGNN_model.parameters()),  # CGNN的参数
#     lr=5e-5  # 学习率
# )

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0.0
    full_loss = 0.0
    finetuned_model.train()
    predictions = []  # 用来存储所有预测文本
    targets = []  # 用来存储所有目标文本

    all_vectors = []
    all_labels = []

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    for step, batch in enumerate(train_loader):  # 直接迭代 DataLoader
        # ------------------ 移动到 GPU ------------------
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        multi_label = batch['multi_label'].to(device)
        # aspect_mask = batch['aspect_mask'].to(device)
        # opinion_mask = batch['opinion_mask'].to(device)

        # tokens, target_text, element_positions, seq_labels, quad_token_list 保持 list 类型
        tokens_list = batch['tokens']
        target_text_list = batch['target_text']

        # element_positions_list = batch['element_positions']
        # seq_labels_list = batch['seq_labels']
        # quad_token_list_list = batch['quad_token_list']

        batch_size = input_ids.size(0)

        ## Step 1: 编码器前向传播
        encoder_outputs = finetuned_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        #T5 编码器输出的隐藏向量
        encoder_hidden = encoder_outputs.last_hidden_state  #torch.Size([1, 512, 768])

        #------------------ 解码器 forward ------------------
        decode_outputs = finetuned_model(
            input_ids=input_ids,  # 解码器的输入仍然是输入的token IDs
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            labels=labels,
            output_hidden_states=True,  # 让模型返回隐藏状态
            return_dict=True
        )

        #获取解码层的隐藏向量
        decoder_last_hidden = decode_outputs.decoder_hidden_states[-1].to(device)  # 形状 (batch, seq_len, hidden_size)

        # 解码预测序列
        logits = decode_outputs.logits
        for i in range(batch_size):
            # 获取预测文本
            predictions_text = tokenizer.decode(logits[i].argmax(dim=-1), skip_special_tokens=True)
            predictions.append(predictions_text)

            # 获取目标文本
            target_text = target_text_list[i]
            targets.append(target_text)

            # # 输出对比
            # print(f"predictions_text: {predictions_text}")
            # print(f"target_text:{target_text}")


        #------------------ 对比学习 ------------------
        # 构建 sentence_vectors 和 all_labels
        sentence_vectors = []
        labels_list = []

        for i in range(batch_size):
            tokens = tokens_list[i]
            pipe_positions = [j for j, token in enumerate(tokens) if token == '▁|']
            hidden_sentence = encoder_hidden[i, :pipe_positions[0], :]  # [seq_len, hidden_dim]
            sentence_vector = hidden_sentence.mean(dim=0)  # [hidden_dim]

            ml = multi_label[i]  # [num_labels]
            has_label = False
            for label_idx, v in enumerate(ml):
                if v == 1:
                    sentence_vectors.append(sentence_vector)
                    labels_list.append(label_idx)
                    has_label = True

                    # ------------------ 保存到 all_vectors / all_labels ------------------
                    all_vectors.append(sentence_vector.detach().cpu().numpy())
                    all_labels.append(label_idx)

        # 如果本 batch 没有正样本，跳过
        if len(sentence_vectors) == 0:
            continue

        vectors = torch.stack(sentence_vectors).to(device)  # [N_samples, hidden_dim]
        labels_tensor = torch.tensor(labels_list, dtype=torch.long).to(device)

        vectors = F.normalize(vectors, dim=1)
        sim_matrix = torch.matmul(vectors, vectors.T) / 0.07

        labels_eq = labels_tensor.unsqueeze(1) == labels_tensor.unsqueeze(0)
        mask = torch.eye(len(vectors), dtype=torch.bool).to(device)
        positive_mask = labels_eq & ~mask

        exp_sim = torch.exp(sim_matrix)
        positive_sum = (exp_sim * positive_mask.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_matrix))
        contrastive_loss = -torch.log(positive_sum / all_sum + 1e-8).mean()
        print(f"Step {step}, Contrastive Loss: {contrastive_loss.item():.4f}")

        #------------------ T5 损失 ------------------
        t5_loss = decode_outputs.loss
        # full_loss += t5_loss
        print(f"Step {step}, T5 Loss: {t5_loss.item():.4f}")

        #Plan B, 将T5微调作为主导，其他的任务使用权重来进行联合训练， 推荐权重为0.2
        total_loss = t5_loss + 0.2 * contrastive_loss

        # # Plan A: 冻结T5解码器。只使用对比学习来优化T5编码器
        # total_loss = contrastive_loss

        full_loss += total_loss
        print(f"Step {step}, Total Loss: {total_loss.item():.4f}")
        print("_______________________________________")

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    # 计算每个 Epoch 的 P/R/F1
    precision, recall, f1 = compute_prf1(predictions, targets)
    avg_loss = full_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch + 1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import numpy as np
    # 转换为 numpy
    X = np.array(all_vectors)
    y = np.array(all_labels)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['EA&EO', 'EA&IO', 'IA&EO', 'IA&IO']

    for i, label in enumerate([0, 1, 2, 3]):
        idx = (y == label)
        if np.any(idx):  # 避免没有该类样本时报错
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                        c=colors[i], label=labels[i], alpha=0.6, s=10)

    plt.legend()
    plt.title("t-SNE Visualization of Sentence Representations")
    plt.savefig(f"D:/Project/Causal_ASQP/Epoch_Figure/Laptop_2epoch{epoch + 1}.png")
    # plt.show()
    plt.close()

    with open("D:/Project/Causal_ASQP/Laptop2.txt", "a", encoding="utf-8") as f:
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}", file=f)


# 保存微调后的模型
output_dir = "D:/Project/Causal_ASQP/ModelPath/LapTop2"
finetuned_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)





