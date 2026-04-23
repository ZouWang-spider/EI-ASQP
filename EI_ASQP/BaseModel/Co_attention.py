import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttention(nn.Module):
    def __init__(self, d_model, max_length=512):
        super(CoAttention, self).__init__()

        # 输入维度
        self.d_model = d_model
        self.max_length = max_length

        # 注意力矩阵（Q: 查询，K: 键，V: 值）
        self.query_fc = nn.Linear(d_model, d_model)
        self.key_fc = nn.Linear(d_model, d_model)
        self.value_fc = nn.Linear(d_model, d_model)

        # Aspect 和 Opinion 的线性变换
        self.aspect_fc = nn.Linear(d_model, d_model)
        self.opinion_fc = nn.Linear(d_model, d_model)

        # 输出层
        self.output_fc = nn.Linear(d_model * 2, d_model)

    def forward(self, hs, aspect_mask=None, opinion_mask=None):
        # Q, K, V
        Q = self.query_fc(hs)  # (n, d)
        K = self.key_fc(hs)
        V = self.value_fc(hs)

        # attention score
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / (self.d_model ** 0.5)  # (n, n)

        # 处理 mask
        mask_matrix = None
        n = hs.size(0)

        if aspect_mask is not None:
            aspect_mask = aspect_mask.squeeze(0).bool()  # (n,)
        if opinion_mask is not None:
            opinion_mask = opinion_mask.squeeze(0).bool()  # (n,)

        if aspect_mask is not None and opinion_mask is not None:
            if aspect_mask.any() and opinion_mask.any():
                mask_matrix = (aspect_mask.unsqueeze(1) & opinion_mask.unsqueeze(0)) | \
                              (opinion_mask.unsqueeze(1) & aspect_mask.unsqueeze(0))  # (n, n)
        elif aspect_mask is not None:
            if aspect_mask.any():
                mask_matrix = aspect_mask.unsqueeze(1) & aspect_mask.unsqueeze(0)
        elif opinion_mask is not None:
            if opinion_mask.any():
                mask_matrix = opinion_mask.unsqueeze(1) & opinion_mask.unsqueeze(0)

        # 应用 mask
        if mask_matrix is not None:
            mask_matrix = mask_matrix.to(attention_scores.device)
            attention_scores = attention_scores.masked_fill(~mask_matrix, float('-inf'))

        # softmax（自动避免 nan）
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # 加权求和
        co_attention_output = torch.matmul(attention_weights, V)

        # aspect / opinion 各自变换
        aspect_rep = self.aspect_fc(co_attention_output)
        opinion_rep = self.opinion_fc(co_attention_output)

        # 拼接输出
        final_representation = torch.cat([aspect_rep, opinion_rep], dim=-1)
        fc_output = self.output_fc(final_representation)

        return fc_output, aspect_rep, opinion_rep



################################  Test  ##############################
# hs = torch.randn(10, 768)  # 假设有10个token，每个token的维度是256
# aspect_mask = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])  # 方面词的mask (假设第1, 4, 9个是方面词)
# opinion_mask = torch.tensor([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])  # 观点词的mask (假设第2, 7个是观点词)
#
# # 初始化 Co-attention 模型
# co_attention_model = CoAttention(d_model=768)
#
# # 使用模型进行前向计算
# fc_output, aspect_rep, opinion_rep = co_attention_model(hs, aspect_mask, opinion_mask)
#
#
# # 输出 Co-attention 的结果
# print("Output Shape:", fc_output)  # (n, d)
# print("Aspect Representation Shape:", aspect_rep.shape)  # (n, d)
# print("Opinion Representation Shape:", opinion_rep.shape)  # (n, d)
