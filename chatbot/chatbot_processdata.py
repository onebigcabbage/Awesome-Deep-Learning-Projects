
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

from chatbot_vocabulary import Voc

PAD_token = 0
SOS_token = 1
EOS_token = 2


# Load & Preprocess Data
# ----------------------
#
# -  220,579 conversational exchanges between 10,292 pairs of movie
#    characters
# -  9,035 characters from 617 movies
# -  304,713 total utterances
#

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# Create formatted data file

'''
{'L1045': {'lineID': 'L1045', 'characterID': 'u0', 'movieID': 'm0',
  'character': 'BIANCA', 'text': 'They do not!\n'}, {}, ...}
'''
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

"""
[{'character1ID': 'u0', 'character2ID': 'u2', 'movieID': 'm0', 'utteranceIDs': "['L198', 'L199']\n"}, 'lines': [{}, {}], ...} 
"""
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # 将str类型的 ['L204', 'L205', 'L206'] 转换成list参与计算
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# 提取对话对
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # 每个句子和下面的那个句子作为一个对话，最后一句话因为没有回答，丢掉
        for i in range(len(conversation["lines"]) - 1):  
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # 避免一个对话中有空句子
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def preSource(datafile, corpus):
    printLines(os.path.join(corpus, "movie_lines.txt"))

    delimiter = '\t'
    # 不知道什么意思
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        # 手动设定 \t 做分隔符(delimiter),\n 结束一行
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    print("\nSample lines from file:")
    printLines(datafile)


# Create vocabulary and dialogual pairs form formatted data file
# 将Unicode类型的string转换成ASCII
# https://stackoverflow.com/a/518232/2809427
# 大写转换为小写，去除非字母和基本符号的其他符号
# 过滤掉句子长度超过预设长度的句子，加快训练收敛
def unicodeToAscii(s):
    return ''.join(
        # 将unicode文本标准化 https://python3-cookbook.readthedocs.io/zh_CN/latest/c02/p09_normalize_unicode_text_to_regexp.html
        c for c in unicodedata.normalize('NFD', s)
        # 返回字符在UNICODE里分类的类型
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    #  r 表示原生字符串，不转义反斜杠
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # 处理空格
    s = re.sub(r"\s+", r" ", s).strip()
    return s

'''
pairs:
[['you re asking me out . that s so cute . what s your name again ?', 'forget it .'],...]
'''
def readVocs(datafile):
    print("Reading lines...")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(datafile)
    return voc, pairs

def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def loadPrepareData(datafile, max_length):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs




######################################################################

# 1) 去除低频词
#
# 2) 去除含有低频词的句子
#

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs




######################################################################
# Prepare Data for Models


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# note： pad到最大长度，按列输出
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    # 按照query长度排序pair，都填充为最长的句子长度
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


