# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


# 两个句子的相似度
def sentence_similarity(sent1, sent2):
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        vector1[all_words.index(w)] += 1
        
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences):
    
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    return similarity_matrix


def generate_summary(file_name, top_n=5):
    
    summarize_text = []
    
    data = open(file_name, "r").read().replace('\n', ' ')
    
    # 切分句子
    article = data.split('. ')
    sentences = []
    
    for sen in article:
        sentences.append(sen.split(" "))
        
    # 句子转化为向量，并计算相似度
    sentence_similarity_matrix = build_similarity_matrix(sentences)
    
    # 得到相似矩阵图
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    print(sentence_similarity_graph.edges(data=True))
    
    # 计算图中每个节点的PageRank值
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # 构建最后结果
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        
    print("Summarize: \n", ". ".join(summarize_text))
    
    
if __name__ == '__main__':
    generate_summary("./data.txt")
    



