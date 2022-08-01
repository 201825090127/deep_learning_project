import spacy
import torch
import collections
import re
import spacy #分词工具
# from nltk.tokenize import word_tokenize #分词工具
# from nltk import data

def read_time_machine():
    with open('./data/time_machine.txt', 'r') as f:
        #lower转为小写
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列，不能删除
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines




#对每一个句子进行分词，将一个句子划分为若干词，转换为一个词的序列
#传入参数
#sentences:一个句子
#token:切分尺度
#返回二维列表
#缺点：无法处理标点符号，
#标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
#类似“shouldn't", "doesn't"这样的词会被错误地处理
#类似"Mr.", "Dr."这样的词会被错误地处理
def tokenize(sentences, token='word'):
    if token=='word':
        return [sentence.split(' ') for sentence in sentences]
    elif token=='char':
        return [list(sentence) for sentence in sentences]
    else:
         print('ERROR: unkown token type '+token)


#建立词典
#先构建一个字典（vocabulary），将每个词映射到一个唯一的索引编号。根据索引编号返回词
class Vocab(object):
    #tokens：二维列表
    #min_freq:阈值。小于这个阈值的去除
    #use_special_tokens：是否要使用特殊字符
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):

        counter = count_corpus(tokens)  # 统计词频，(key,value)==>(词，词的数量)
        self.token_freqs = list(counter.items())
        #词性列表
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            #batch的长度不一定一样，在短的句子补上<pad>
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk = 0
            self.idx_to_token += ['<unk>']
        self.idx_to_token += [token for token, freq in self.token_freqs  #token_freqs （词，词频）
                        if freq >= min_freq and token not in self.idx_to_token]

        #构建词到索引的映射
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    #给定词，返回对应的索引
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens] #递归


    #给定索引，返回对应的词
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(index) for index in indices] #递归

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]  # 二维列表展平
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数

if __name__=="__main__":
    lines = read_time_machine()
    print('# sentences %d' % len(lines))
    print(lines[0])
    tokens=tokenize(lines)
    mp=count_corpus(tokens)
    print(mp['the'])
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[0:10])#将dict_item转为list
    print(vocab.to_tokens([[5,4],[4]]))
    print(vocab.token_to_idx.items())
    print(vocab.__getitem__([['by','g'],['my']]))
    print(vocab.idx_to_token)
