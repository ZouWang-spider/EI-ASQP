import torch
from Causal_ASQP.BaseModel.Token_Process import Tokenizer_Label

def merge_tokens(tokens):
    merged_tokens = []
    original_positions = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith('▁'):
            merged_tokens.append(token + tokens[i + 1])
            original_positions.append([i, i + 1])
            i += 2
        else:
            merged_tokens.append(token)
            original_positions.append([i])
            i += 1
    return merged_tokens, original_positions


def construct_quad_token_matrix(aspect_mask, opinion_mask, quad_label, seq_len):

    # 初始化矩阵，4行 (A,O,C,S)，列为句子长度，全部填 N
    quad_token_matrix = [['N'] * seq_len for _ in range(4)]

    # 四元组元素
    aspect, opinion, category, sentiment = quad_label[0]  # 假设单个四元组
    aspect_exists = aspect and aspect != 'NULL'
    opinion_exists = opinion and opinion != 'NULL'

    # A 行
    if aspect_exists:
        for i, val in enumerate(aspect_mask):
            if val == 1:
                quad_token_matrix[0][i] = 'A'
        # C 行与A对应
        for i, val in enumerate(aspect_mask):
            if val == 1:
                quad_token_matrix[2][i] = 'C'
    else:
        quad_token_matrix[0] = ['T'] * seq_len
        quad_token_matrix[2] = ['N'] * seq_len  # 类别与句子token不对应

    # O 行
    if opinion_exists:
        for i, val in enumerate(opinion_mask):
            if val == 1:
                quad_token_matrix[1][i] = 'O'
        # S 行与O对应
        for i, val in enumerate(opinion_mask):
            if val == 1:
                quad_token_matrix[3][i] = 'S'
    else:
        quad_token_matrix[1] = ['T'] * seq_len
        quad_token_matrix[3] = ['N'] * seq_len  # 情感与句子token不对应

    return quad_token_matrix

def get_quad_token_list(tokens, quad_label):
    all_quad_matrices = []
    for quad in quad_label:
        # 1. 针对当前四元组生成 aspect_mask, opinion_mask, seq_labels
        aspect_mask, opinion_mask, seq_labels = Tokenizer_Label(tokens, [quad])

        # 2. 构建当前四元组的 quad-token 矩阵 (4, n)
        quad_matrix = construct_quad_token_matrix(aspect_mask, opinion_mask, [quad], len(aspect_mask))

        all_quad_matrices.append(quad_matrix)

    merged_list = [row for quad_matrix in all_quad_matrices for row in quad_matrix]

    matrix_label2id = {"N": 0, "A": 1, "O": 2, "C": 3, "S": 4, "T": 5}

    # 假设 merged_list 是一个二维列表
    quad_token_list = [
        [matrix_label2id[label] for label in row]
        for row in merged_list
    ]
    return quad_token_list





import torch.nn.functional as F

# 计算余弦相似度
def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def calculate_quad_token_loss(hidden_sentence, quad_outputs, quad_token_list):
    # quad_token_matrix 是你提前定义好的真实标签矩阵
    quad_token_matrix = torch.tensor(quad_token_list, dtype=torch.float)  # shape: [batch_size, num_tokens]

    similarities = []
    for i, quad_output in enumerate(quad_outputs):
        for j in range(hidden_sentence.shape[0]):  # 遍历每个 token
            similarity = cosine_similarity(hidden_sentence[j], quad_output.squeeze(0))
            similarities.append(similarity)

    # 展平所有相似度
    flattened_similarities = [sim.item() if sim.dim() == 0 else sim.mean().item() for sim in similarities]

    # Ensure pred_labels are of type float (logits)
    pred_labels = torch.tensor(flattened_similarities).view(-1, quad_token_matrix.shape[1]).float()


    return pred_labels, quad_token_matrix




######################   Test   ######################
# tokens = ['▁next', '▁', ',', '▁is', '▁that', '▁the', '▁track', '▁pad', '▁is', '▁insane', 'ly', '▁wo', 'b', 'b', 'ly', '▁', '.', '▁|', 'ly', '▁', '.', 'ly', '▁', '.']
# quad_label = [['track pad', 'wobbly', 'hardware operation_performance', 'negative'], ['next', 'NULL', 'Laptop quality', 'negative']]
# quad_label = [['track pad', 'wobbly', 'hardware operation_performance', 'negative']]

# aspect_mask, opinion_mask, seq_labels = Tokenizer_Label(tokens, quad_label)
# print(aspect_mask)
# print(opinion_mask)
#
# quad_matrix = construct_quad_token_matrix(aspect_mask, opinion_mask, quad_label, len(aspect_mask))

# quad_token_list = get_quad_token_list(tokens, quad_label)
#
# for row in quad_token_list:
#     print(row)


# # 假设 quad_outputs 和 hidden_sentence 都已经计算好
# hidden_sentence = torch.randn(9, 768)  # 句子的 9 个 token 特征
# quad_outputs = [torch.randn(1, 768), torch.randn(1, 768), torch.randn(2, 768), torch.randn(1, 768)]  # 情感四元组特征
#
# # quad_token_matrix 是你提前定义好的真实标签矩阵
# quad_token_list = [
#     [5, 5, 5, 5, 5, 5, 5, 5, 5],  # 情感四元组1
#     [5, 5, 5, 5, 5, 5, 5, 5, 5],  # 情感四元组2
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 情感四元组3
#     [0, 0, 0, 0, 0, 0, 0, 0, 0]   # 情感四元组4
# ]
#
# # 计算损失
# pred_labels, quad_token_matrix = calculate_quad_token_loss(hidden_sentence, quad_outputs, quad_token_list)
