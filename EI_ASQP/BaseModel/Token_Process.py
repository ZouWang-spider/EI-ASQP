import torch


def merge_tokens(tokens):
    """
    Merge sub-tokens into words based on the presence of the '▁' prefix.
    Also returns the original positions of the tokens that were merged.
    """
    merged_tokens = []
    original_positions = []  # 用于存储每个合并后的单词在原始token中的位置
    current_token = ""
    current_positions = []  # 当前单词的位置
    is_merging = False  # 标志位，判断是否正在合并子词

    for idx, token in enumerate(tokens):
        if token.startswith('▁'):  # Token starts a new word
            # 如果已经有了一个合并的单词，先把它加到结果中
            if current_token:
                merged_tokens.append(current_token)
                original_positions.append(current_positions)
            # 以当前的token开始新的合并
            current_token = token[1:]  # 去掉'▁'，开始一个新的单词
            current_positions = [idx]  # 记录当前token的原始位置
            is_merging = True
        elif token == '▁':  # 空格符号
            if current_token:
                merged_tokens.append(current_token)
                original_positions.append(current_positions)
            current_token = ""  # 重置为新单词
            current_positions = []  # 重置位置
            is_merging = False
        else:
            # 如果是拆分的子词，继续合并
            if is_merging:
                current_token += token  # 拼接子词
                current_positions.append(idx)  # 添加当前token的位置

    # 如果当前token还未加入（避免循环结束时遗漏最后一个合并的单词）
    if current_token:
        merged_tokens.append(current_token)
        original_positions.append(current_positions)

    return merged_tokens, original_positions


# 定义映射
label2id = {"N": 0, "A": 1, "O": 2}

def Tokenizer_Label(full_tokens, reordered_quads):
    # 提取句子部分，截取'▁|'前的token
    pipe_positions = [i for i, token in enumerate(full_tokens) if token == '▁|']
    tokens = full_tokens[:pipe_positions[0]]

    # 初始化 aspect_mask、opinion_mask 和 seq_labels
    aspect_mask = torch.zeros(len(tokens), dtype=torch.int32)
    opinion_mask = torch.zeros(len(tokens), dtype=torch.int32)
    seq_labels = ['N'] * len(tokens)

    # 合并拆分的token，并获取合并后的token和原始token的位置
    merged_tokens, original_positions = merge_tokens(tokens)

    # 遍历每个四元组
    for quad in reordered_quads:
        aspect = quad[0]  # 方面词
        opinion = quad[1]  # 观点词

        # 处理方面词
        if aspect and aspect != 'NULL':
            aspect_tokens = aspect.split()  # 拆分多词
            for token in aspect_tokens:
                for i, t in enumerate(merged_tokens):
                    if token.lower() == t.lower():
                        aspect_positions = original_positions[i]
                        for pos in aspect_positions:
                            aspect_mask[pos] = 1
                        seq_labels[aspect_positions[0]] = 'A'  # 标记开始位置

        # 处理观点词
        if opinion and opinion != 'NULL':
            opinion_tokens = opinion.split()
            for token in opinion_tokens:
                for i, t in enumerate(merged_tokens):
                    if token.lower() == t.lower():
                        opinion_positions = original_positions[i]
                        for pos in opinion_positions:
                            opinion_mask[pos] = 1
                            if seq_labels[pos] == 'N':
                                seq_labels[pos] = 'O'

    seq_labels = [label2id[label] for label in seq_labels]

    return aspect_mask, opinion_mask, seq_labels




##############################    Test    ##############################################
# tokens = ['▁next', '▁', ',', '▁is', '▁that', '▁the', '▁track', '▁pad', '▁is', '▁insane', 'ly', '▁wo', 'b', 'b', 'ly', '▁', '.', '▁|', 'ly', '▁', '.', 'ly', '▁', '.']
# quad_label = [['track pad', 'wobbly', 'hardware operation_performance', 'negative'], ['next', 'NULL', 'Laptop quality', 'negative']]  # 示例情感四元组
#
# # 调用函数获取标签
# aspect_mask, opinion_mask, seq_labels = Tokenizer_Label(tokens, quad_label)
# # 打印结果
# print("Aspect Mask:", aspect_mask)
# print("Opinion Mask:", opinion_mask)
# print("Seq Labels:", seq_labels)


