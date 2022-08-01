import spacy
import torch
import collections
import re


with open('./data/jaychou_lyrics.txt',encoding='utf-8') as f:
    corpus_chars = f.read()
print(len(corpus_chars))
print(corpus_chars[: 10])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[: 10000]

print(corpus_chars[:10])

idx_to_char=list(set(corpus_chars))# 去重，得到索引到字符的映射
char_to_idx={char:i for i,char in enumerate(idx_to_char)}#得到字符到索引的映射
vocab_size=len(char_to_idx)
print(char_to_idx)
corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 将文本根据char_to_idx转为索引列表
sample = corpus_indices[: 20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)

import torch
import random

#一个序列包含很多样本，一个样本就是一个batch_size，样本长度为
#
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符
    num_examples = (len(corpus_indices) - 1) // num_steps  # 下取整，得到不重叠情况下的样本个数
    example_indices = [i * num_steps for i in range(num_examples)]  # 每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices)

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps] #左闭右开区间

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(0, num_examples, batch_size):
        print(i)
        # 每次选出batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices] #列表推导式 用来专门生成列表，比二维循环和append简单快速
        Y = [_data(j + 1) for j in batch_indices] #
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)

my_seq=range(30)
batch=0
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print("第{}个batch".format(batch))
    print('X: ', X, '\nY:', Y)
    print("=================================")
    batch+=1

