import os
import pickle
import pandas as pd

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

df_file = pd.read_pickle("dataset/df_final.pkl")
temp = df_file.loc[:,'impression'].str.replace(".","").copy() #removing all fullstops and storing the result in a temp variable
words = []
for i in temp.values:
    k = i.split()
    for word in k:
        if word not in words:
            words.append(word)
vocab = Vocabulary()
vocab.add_word('<pad>') # 0
vocab.add_word('<start>') # 1
vocab.add_word('<end>') # 2
vocab.add_word('<unk>') # 3
for i, word in enumerate(words):
    vocab.add_word(word)

with open("dataset/vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)


 