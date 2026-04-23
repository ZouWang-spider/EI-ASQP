import torch

#从目标文件中获取情感四元素的位置信息
def extract_quad_positions(target_text):
    # 按空格拆分为 token
    tokens = target_text.split()
    # 记录结果
    quads = []
    # 每个四元组用这个字典保存
    current_quad = {}

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]

        if token == '[A]':
            start = idx + 1
            # 找到下一个标签前的位置
            end = start
            while end < len(tokens) and not tokens[end].startswith('['):
                end += 1
            current_quad['A'] = (start, end - 1)
            idx = end
            continue

        elif token == '[O]':
            start = idx + 1
            end = start
            while end < len(tokens) and not tokens[end].startswith('['):
                end += 1
            current_quad['O'] = (start, end - 1)
            idx = end
            continue

        elif token == '[C]':
            start = idx + 1
            end = start
            while end < len(tokens) and not tokens[end].startswith('['):
                end += 1
            current_quad['C'] = (start, end - 1)
            idx = end
            continue

        elif token == '[S]':
            start = idx + 1
            end = start
            while end < len(tokens) and not tokens[end].startswith('['):
                end += 1
            current_quad['S'] = (start, end - 1)
            idx = end
            continue

        elif token == '[SSEP]':
            # 遇到分割符，当前四元组存入结果
            if current_quad:
                quads.append(current_quad)
            current_quad = {}
            idx += 1
            continue

        else:
            idx += 1

    # 处理最后一个四元组
    if current_quad:
        quads.append(current_quad)

    return tokens, quads

#######################   Test1   ###########################
# target_text = "[A] computer [O] love [C] laptop general [S] positive [SSEP] [A] tablet device [O] love [C] laptop general [S] positive"
# tokens, positions = extract_quad_positions(target_text)
#
# print("Tokens:", tokens)
# print("Positions:", positions)


# 假设 decoder_last_hidden 是形状 [1, seq_len, embed_dim]
# element_positions 是包含多个四元组位置的列表
# CGNN_model 是你已经定义好的因果生成模型
def extract_element_hidden(decoder_hidden, positions):

    hidden_vectors = {}
    for key, (start, end) in positions.items():
        # slice 对应token的隐藏向量，end 是包含的结束位置
        # 注意decoder_hidden形状是(1, seq_len, embed_dim)
        # 取第0个batch
        vecs = decoder_hidden[0, start:end + 1, :]  # (token_count, embed_dim)
        # 取平均作为该元素的向量表示
        hidden_vectors[key] = vecs.mean(dim=0)  # (embed_dim,)
    return hidden_vectors


# 你已定义好的CGNN模型
def CGNN_Compute(element_positions,decoder_last_hidden,  CGNN_model):

    all_outputs = []

    for quad_pos in element_positions:
        # 从 decoder_last_hidden 中提取四元素，保留多词维度
        E1 = decoder_last_hidden[0, quad_pos['A'][0]:quad_pos['A'][1] + 1, :]  # (num_tokens_A, 768)
        E2 = decoder_last_hidden[0, quad_pos['O'][0]:quad_pos['O'][1] + 1, :]  # (num_tokens_O, 768)
        E3 = decoder_last_hidden[0, quad_pos['C'][0]:quad_pos['C'][1] + 1, :]  # (num_tokens_C, 768)
        E4 = decoder_last_hidden[0, quad_pos['S'][0]:quad_pos['S'][1] + 1, :]  # (num_tokens_S, 768)
        # print(E1.shape)
        # print(E2.shape)
        # print(E3.shape)
        # print(E4.shape)
        # print('--------------')

        # 通过 CGNN 计算输出
        x1, x2, x3, x4 = CGNN_model(E1, E2, E3, E4)

        # 按顺序存入 all_outputs
        all_outputs.extend([x1, x2, x3, x4])  # 不用元组，依次追加
    return all_outputs


#######################  Test2  ######################
# from Causal_ASQP.BaseModel.CGNN import CausalGenerationModel
#
# CGNN_model = CausalGenerationModel(embed_dim=768, hidden_dim=512)
#
# decoder_last_hidden = torch.randn(1, 128, 768)  # 示例张量
#
# element_positions = [
#     {'A': (1, 1), 'O': (3, 3), 'C': (5, 6), 'S': (8, 8)},
#     {'A': (9, 12), 'O': (14, 14), 'C': (16, 17), 'S': (19, 19)}
# ]
#
# all_outputs = CGNN_Compute(element_positions,decoder_last_hidden,  CGNN_model)
#
# # 查看all_outputs的长度（四元组个数）
# print("Number of quads:", len(all_outputs))
#
# # 查看每个四元组的元素及其维度
# for i, quad in enumerate(all_outputs):
#     print(quad.shape)


