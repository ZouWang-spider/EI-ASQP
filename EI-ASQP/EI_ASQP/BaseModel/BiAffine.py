import nltk
import dgl
import torch
import networkx as nx
import torch.nn as nn
from supar import Parser
from transformers import BertTokenizer, BertModel


#使用BiAffine对句子进行处理得到arcs、rels、probs
def BiAffine(sentence):
      tokens = nltk.word_tokenize(sentence)

      # 获取词性标注
      ann = nltk.pos_tag(tokens)
      pos_tags = [pair[1] for pair in ann]


      parser = Parser.load('D:/BiAffine/ptb.biaffine.dep.lstm.char')  # 'biaffine-dep-roberta-en'解析结果更准确
      dataset = parser.predict([tokens], prob=True, verbose=True)

      #dependency feature
      # rels = dataset.rels[0]
      # print(f"arcs:  {dataset.arcs[0]}\n"
      #       f"rels:  {dataset.rels[0]}\n"
      #       f"probs: {dataset.probs[0].gather(1, torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")

      # 构建句子的图，由弧-->节点
      arcs = dataset.arcs[0]  # 边的信息
      edges = [i + 1 for i in range(len(arcs))]
      for i in range(len(arcs)):
            if arcs[i] == 0:
                  arcs[i] = edges[i]

      # 将节点的序号减一，以便适应DGL graph从0序号开始
      arcs = [arc - 1 for arc in arcs]
      edges = [edge - 1 for edge in edges]
      graph = (arcs, edges)


      # Create a DGL graph
      text_graph = torch.tensor(graph)

      # 创建一个有权图
      G = nx.Graph()
      for i, j in zip(arcs, edges):
            G.add_edge(i, j, weight=1)

      # 计算词性-词性边关系
      pos2pos_edges = []
      for head_idx, dep_idx in zip(arcs, edges):
            head_pos = pos_tags[head_idx]
            dep_pos = pos_tags[dep_idx]
            pos2pos_edges.append(f"{head_pos}-{dep_pos}")

      return tokens, text_graph, G, pos2pos_edges


# 加载BERT模型和分词器
model_name = 'D:/bert-base-cased'  # 您可以选择其他预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

#实现边特征的嵌入功能，这里的边为（POS-POS）作为边的特征
def BERT_Embedding(tokens, pos2pos_edges):

      # 获取单词节点特征
      marked_text1 = ["[CLS]"] + tokens + ["[SEP]"]
      marked_text3 = ["[CLS]"] + pos2pos_edges + ["[SEP]"]

      # 将分词转化为词向量 word embedding
      input_ids1 = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
      outputs1 = model(input_ids1)

      # pos_pos作为边的关系权重
      input_ids = torch.tensor(tokenizer.encode(marked_text3, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
      outputs = model(input_ids)

      # 获取词向量
      word_embeddings = outputs1.last_hidden_state
      # 获取词向量
      pos_embeddings = outputs.last_hidden_state

      # 提取单词对应的词向量（去掉特殊标记的部分）
      word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
      # 提取单词对应的词向量（去掉特殊标记的部分）
      pos_embeddings = pos_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记

      # 使用切片操作去除第一个和最后一个元素
      word_feature = word_embeddings[0][1:-1, :]  # 单词特征
      pos_pos_feature = pos_embeddings[0][1:-1, :]  # 单词特征

      return word_feature, pos_pos_feature




############################################## Test ###########################################
# sentence = "I will be going back very soon"
# tokens, text_graph, G, pos2pos_edges = BiAffine(sentence)
# print(text_graph)
# print(pos2pos_edges)
#
# word_feature, pos_pos_feature = BERT_Embedding(tokens, pos2pos_edges)
# print(word_feature.shape)
# print(pos_pos_feature.shape)

