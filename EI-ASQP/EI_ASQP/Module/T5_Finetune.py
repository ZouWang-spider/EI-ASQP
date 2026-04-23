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



# 选择模型（使用 't5-small', 't5-based','t5-large', 但需更多显存 't5-3b', 't5-11b'）
tokenizer = T5Tokenizer.from_pretrained("D:\T5_Based")
t5model = T5ForConditionalGeneration.from_pretrained("D:\T5_Based")


# 设置输入句子（T5 要求输入带任务提示）
# 2. 配置 AdaLoRA（你可以根据硬件和精度需求自定义）
adalo_config = AdaLoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,   # 对于 T5 是序列到序列
    r=8,                               # 降维维度（通常为 4~16）
    target_modules=["q", "v"],         # 针对注意力的 query 和 value 层
    lora_alpha=32,                     # 缩放因子
    lora_dropout=0.1,                  # Dropout 用于防止过拟合
    # init_r=6,
    # beta1=0.85,
    # beta2=0.85,
    # tinit=100,
    # tfinal=1000,
    # delta_t=10,
)

# 3. 将模型转换为 AdaLoRA 模式
finetuned_model = get_peft_model(t5model, adalo_config)
finetuned_model.print_trainable_parameters()


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


# ---------- T5 模型的微调训练开始 ----------
file_path = r"D:\Project\Causal_ASQP\Dataset\Restaurant\train.txt"
train_dataset = QuadDataset(file_path, tokenizer)


# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.is_available())
finetuned_model.to(device)

# # 初始化 Co-attention 模型
# co_attention_model = CoAttention(d_model=768).to(device)
#
# BCE_criterion = nn.BCEWithLogitsLoss().to(device)

# #多标签分类任务
# fc_layer = nn.Sequential(
#     nn.Linear(768, 256),
#     nn.ReLU(),
#     nn.Linear(256, 4)
# ).to(device)
#
# #CRF序列标注
# crf = CRF(3, batch_first=True).to(device)
# linear = nn.Sequential(
#     nn.Linear(768, 256),  # 第一层
#     nn.ReLU(),            # 激活函数
#     nn.Linear(256, 3)     # 第二层
# ).to(device)
#
# #因果生成神经网络初始化 CGNN
# CGNN_model = CausalGenerationModel(embed_dim=768, hidden_dim=512).to(device)
#
#
# #CE 损失
# CE_fn = torch.nn.CrossEntropyLoss().to(device)

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
    for step, idx in enumerate(torch.randperm(len(train_dataset))):  # 打乱顺序
        batch = train_dataset[idx]

        input_text = batch['input_text']
        tokens = batch['tokens']
        input_ids = batch['input_ids'].unsqueeze(0).to(device)  # 添加 batch 维度
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
        labels = batch['labels'].unsqueeze(0).to(device)
        target_text = batch['target_text']
        element_positions = batch['element_positions']

        # 提取 aspect_mask 和 opinion_mask
        aspect_mask = batch['aspect_mask'].unsqueeze(0).to(device)
        opinion_mask = batch['opinion_mask'].unsqueeze(0).to(device)

        multi_label = batch['multi_label']  # 四分类多标签
        seq_labels = batch['seq_labels']  # 方面-观点联合标注

        quad_token_list = batch['quad_token_list']   #quda-token标签


        ## Step 1: 编码器前向传播
        encoder_outputs = finetuned_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # #T5 编码器输出的隐藏向量
        # encoder_hidden = encoder_outputs.last_hidden_state  #torch.Size([1, 512, 768])
        #
        # # 查找 '▁|' 符号的位置
        # pipe_positions = [i for i, token in enumerate(tokens) if token == '▁|']
        # # 根据 '▁|' 符号的位置提取对应句子的隐藏向量
        # hidden_sentence = encoder_hidden[0, :pipe_positions[0], :].to(device)  # 提取所有'▁|'前的向量
        # # print('hidden_sentence', hidden_sentence.shape)
        #
        # # 句子向量：取隐藏向量的均值
        # sentence_vector = hidden_sentence.mean(dim=0).detach().cpu().numpy()
        #
        # # 保存向量 & 标签
        # # multi_label 是 [0,1,0,0] 形式
        # if isinstance(multi_label, list):
        #     multi_label = np.array(multi_label)
        #
        # # 遍历多标签，把句子拆成多个点
        # for i, v in enumerate(multi_label):
        #     if v == 1:
        #         all_vectors.append(sentence_vector)
        #         all_labels.append(i)  # i 就是类别编号 (0~3)


        #######################  aspect-opinion aware  #############################
        # # 使用Co-attention计算方面词和观点词的交互表示
        # concat_output, aspect_rep, opinion_rep = co_attention_model(hidden_sentence, aspect_mask, opinion_mask)
        #
        # # 池化为句子特征，比如用平均池化
        # sentence_feature = concat_output.mean(dim=0, keepdim=True).to(device)  # (1, 768)
        # implicit_logits = fc_layer(sentence_feature)  # (n,4)
        #
        # # 计算多标签损失EA&EO、 EA&IO、 IA&EO、 IA&IO标签为例如[1, 0, 1, 0]
        # multi_labels = torch.tensor(multi_label, dtype=torch.float32).unsqueeze(0).to(device)  # [1,4]
        # loss_explicit_implicit = BCE_criterion(implicit_logits, multi_labels)
        # # print("loss_explicit_implicit:", loss_explicit_implicit)

        #方面-观点联合序列抽取 label={B-A, I-A, B-O, I-O, N}/ {A, O, N}
        # emissions = linear(hidden_sentence).to(device)  # 对输出进行线性变换
        # emissions = emissions.unsqueeze(0).to(device)  # 添加序列维度 (batch_size, seq_len, num_labels)
        #
        # # 使用CRF进行序列标注任务
        # seq_labels_tensor = torch.tensor([seq_labels], dtype=torch.long).to(device)
        # # 计算 CRF 损失（注意：CRF 的 loss 是负 log-likelihood，需要取反）
        # loss_aspect_opinion = -crf(emissions, seq_labels_tensor, reduction='mean')
        # print('loss_aspect_opinion:', loss_aspect_opinion)

        # Step 6: 使用融合结果送入 decoder
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


        ##################### Causal Optimization Generative ############################
        # #使用CGNN计算得到情感四元组的特征向量，并存储在列表中
        # quad_outputs = CGNN_Compute(element_positions, decoder_last_hidden, CGNN_model)
        #
        # #使用quad特征向量 quad_outputs 与 token的特征向量 hidden_sentence 计算quad-token矩阵
        # pred_labels, quad_token_matrix = calculate_quad_token_loss(hidden_sentence, quad_outputs, quad_token_list)
        # loss_quad = CE_fn(pred_labels, quad_token_matrix)
        # # print(f"loss_quad, {loss_quad}")


        # 解码预测序列
        logits = decode_outputs.logits
        predictions_text = tokenizer.decode(logits.argmax(dim=-1).squeeze(), skip_special_tokens=True)

        # 保存预测和目标文本
        predictions.append(predictions_text)
        targets.append(target_text)
        print('predictions_text:', predictions_text)
        print('target_text:', target_text)

        #T5编码器-解码器计算的损失
        t5_loss = decode_outputs.loss
        # print(f"t5_loss, {t5_loss}")

        # 联合损失返回
        # loss_quad = torch.tensor(loss_quad, dtype=torch.float32).to(device)  # 转换为 Tensor
        # loss_t5 = torch.tensor(t5_loss, dtype=torch.float32).to(device)  # 转换为 Tensor
        # # 联合损失返回
        # total_loss = 0.9 * loss_t5 + 0.08 * loss_explicit_implicit + 0.01 * loss_aspect_opinion + 0.01 * loss_quad
        # total_loss += t5_loss.item()

        full_loss += t5_loss.item()

        print(f"Step {step}, Loss: {t5_loss.item():.4f}")

        t5_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    # 计算每个 Epoch 的 P/R/F1
    precision, recall, f1 = compute_prf1(predictions, targets)

    avg_loss = full_loss / len(train_dataset)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    # print(f"Epoch {epoch + 1} Accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch + 1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE
    # import numpy as np
    # # 转换为 numpy
    # X = np.array(all_vectors)
    # y = np.array(all_labels)
    #
    # # t-SNE
    # tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    # X_tsne = tsne.fit_transform(X)
    #
    # # 绘图
    # plt.figure(figsize=(8, 6))
    # colors = ['red', 'blue', 'green', 'orange']
    # labels = ['EA&EO', 'EA&IO', 'IA&EO', 'IA&IO']
    #
    # for i, label in enumerate([0, 1, 2, 3]):
    #     idx = (y == label)
    #     if np.any(idx):  # 避免没有该类样本时报错
    #         plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
    #                     c=colors[i], label=labels[i], alpha=0.6, s=10)
    #
    # plt.legend()
    # plt.title("t-SNE Visualization of Sentence Representations")
    # plt.savefig(f"D:/Project/Causal_ASQP/Epoch_Figure/Res_epoch{epoch + 1}.png")
    # # plt.show()
    # plt.close()

    with open("D:/Project/Causal_ASQP/T5_Large_Laptop2.txt", "a", encoding="utf-8") as f:
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}", file=f)


# 保存微调后的模型
output_dir = "D:/Project/Causal_ASQP/ModelPath/Laptop_T5_Large2"
finetuned_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
