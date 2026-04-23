import re
import ast
import nltk
import torch
import numpy as np

def normalize_contractions(text):
    # 替换常见英语缩写为完整形式
    contractions = {
        r"\' m\b": "am",
        r"\' re\b": "are",
        r"\' ve\b": "have",
        # r"n\' t\b": "not",
        r"\' ll\b": "will",
        r"\' d\b": "would",
        r"\' s\b": "is",  # 注意：这也可能是所有格's，根据任务可以选择去掉这条
        r"’ m\b": "am",  # smart quote versions
        r"’ re\b": "are",
        r"’ ve\b": "have",
        r"didn ' t\b": "did not",
        r"isn ' t\b": "is not",
        r"don ' t\b": "do not",
        r"wouldn ' t\b": "would not",
        r"can ' t\b": "can not",
        r"doesn ' t\b": "does not",
        r"wasn ' t\b": "was not",
        r"n ' t\b": " not",
        r"ca n ' t\b": "can not",
        r"shouldn ' t\b": "should not",
        r"haven ' t": "have not",
        r"' t\b": "not",
        r"’ ll\b": "will",
        r"’ d\b": "would",
        r"' s\b": "is",
    }

    for pattern, repl in contractions.items():
        text = re.sub(pattern, repl, text)
    return text


def process_dataset(file_path):

    dataset = []
    aspect_opinion_label =[]
    ssep_token = '[SSEP]'


    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # 跳过空行

            try:
                sentence_part, quads_part = line.strip().split("####")

                sentence = sentence_part.strip()
                sentence = normalize_contractions(sentence)
                # quads_raw = ast.literal_eval(quads_part.strip())  # 将字符串转换为真实列表
                quads_raw = ast.literal_eval(normalize_contractions(quads_part.strip()))  # 将字符串转换为真实列表

                reordered_quads = []
                target_spans = []

                # 初始化显-隐组合矩阵: 行是 aspect (隐式0 / 显式1), 列是 opinion (隐式0 / 显式1)
                label_matrix = [[0, 0], [0, 0]]

                # 使用简单的分词器（你可以替换为你使用的T5分词器）
                tokens = nltk.word_tokenize(sentence)
                seq_labels = ['N'] * len(tokens)  # 初始全为'N'

                # 初始化 aspect_mask 和 opinion_mask
                aspect_mask = torch.zeros(len(tokens), dtype=torch.int)
                opinion_mask = torch.zeros(len(tokens), dtype=torch.int)

                for quad in quads_raw:
                    aspect, category, sentiment, opinion = quad

                    #隐式方面词的-二分类标签构造 2×2 方面-观点显-隐矩阵构造
                    aspect_flag = 1 if aspect.strip().lower() == "null" else 0
                    opinion_flag = 1 if opinion.strip().lower() == "null" else 0
                    # print(aspect_flag, opinion_flag)

                    # 更新矩阵标签
                    label_matrix[aspect_flag][opinion_flag] = 1

                    # 将二维矩阵转为一维多标签列表
                    multi_label = [
                        label_matrix[0][0],  # EA&EO
                        label_matrix[0][1],  # EA&IO
                        label_matrix[1][0],  # IA&EO
                        label_matrix[1][1],  # IA&IO
                    ]

                    reordered = [aspect, opinion, category, sentiment]
                    reordered_quads.append(reordered)


                    target_str = f"[A] {aspect} [O] {opinion} [C] {category} [S] {sentiment}"
                    target_spans.append(target_str)


                target_text = f" {ssep_token} ".join(target_spans)

                # 添加到data列表
                dataset.append((sentence, tokens, reordered_quads, target_text))

                #aspect_mask, opinion_mask, seq_labels的序列标签需要依据T5的tokenizer来构造
                aspect_opinion_label.append((multi_label))

            except Exception as e:
                print(f"跳过错误行: {line.strip()}，错误：{e}")

    return dataset, aspect_opinion_label




#构建 quad-token 标签矩阵,标签类别: {A, O, C, S, T, N}
def build_quad_token_matrix(tokens, quads):
    """
    构建 Quad-Token 矩阵
    tokens: 句子分词列表
    quads: 四元组列表 [[aspect, opinion, category, sentiment], ...]
    """
    labels = ['N', 'A', 'O', 'C', 'S', 'T']
    label2id = {l: i for i, l in enumerate(labels)}

    num_tokens = len(tokens)
    num_rows = len(quads) * 4  # 每个四元组对应4行
    matrix = np.full((num_rows, num_tokens), label2id['N'], dtype=int)

    for idx, (aspect, opinion, category, sentiment) in enumerate(quads):
        aspect_null = (aspect.strip().lower() == "null")
        opinion_null = (opinion.strip().lower() == "null")

        # 四元组的四行起始位置
        row_A = idx * 4
        row_O = idx * 4 + 1
        row_C = idx * 4 + 2
        row_S = idx * 4 + 3

        # 找 aspect token 下标
        aspect_pos = []
        if not aspect_null:
            aspect_tokens = aspect.strip().split()
            for i in range(num_tokens - len(aspect_tokens) + 1):
                if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                    aspect_pos = list(range(i, i+len(aspect_tokens)))
                    break

        # 找 opinion token 下标
        opinion_pos = []
        if not opinion_null:
            opinion_tokens = opinion.strip().split()
            for i in range(num_tokens - len(opinion_tokens) + 1):
                if tokens[i:i+len(opinion_tokens)] == opinion_tokens:
                    opinion_pos = list(range(i, i+len(opinion_tokens)))
                    break

        # 方面词为 NULL，方面词行和类别行设为 T
        if aspect_null:
            matrix[row_A, :] = label2id['T']
            matrix[row_C, :] = label2id['N']

        # 观点词为 NULL，观点词行和情感行设为 T
        if opinion_null:
            matrix[row_O, :] = label2id['T']
            matrix[row_S, :] = label2id['N']

        # 行A：方面词位置标记 A
        for pos in aspect_pos:
            matrix[row_A, pos] = label2id['A']

        # 行O：观点词位置标记 O
        for pos in opinion_pos:
            matrix[row_O, pos] = label2id['O']

        # 行C：类别位置标记在方面词位置
        for pos in aspect_pos:
            matrix[row_C, pos] = label2id['C']

        # 行S：情感位置标记在观点词位置
        for pos in opinion_pos:
            matrix[row_S, pos] = label2id['S']

    return matrix, label2id








################################  Test ################################
# file_path = r"D:\Project\Causal_ASQP\Dataset\Laptop\train.txt"
# dataset, aspect_opinion_label= process_dataset(file_path)
#
# for sentence, tokens, reordered_quads, target_text in dataset:
#     print(sentence)
#     print(tokens)
#     print(reordered_quads)
#     print(target_text)
#     quad_token_matrix, label2id = build_quad_token_matrix(tokens, reordered_quads)
#     print("标签映射:", label2id)
#     print(quad_token_matrix)
#     print('____')
#
#
# # multi_label为的四分类多标签 EA&EO、 EA&IO、 IA&EO、 IA&IO
# for multi_label in aspect_opinion_label:
#     print(multi_label)



# tokens = ['the', 'keyboard', 'is', 'nice', 'to', 'type', 'on', '.']
# quads = [['keyboard', 'nice', 'keyboard operation_performance', 'positive']]
# matrix, label2id = build_quad_token_matrix(tokens, quads)
# print("标签映射:", label2id)
# print(matrix)




