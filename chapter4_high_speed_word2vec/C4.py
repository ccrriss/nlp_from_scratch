from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    def forward(self, idx):
        W, = self.params 
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout: np.ndarray):
        dw, = self.grads
        dw[...] = 0
        
        # for idx, word_id in enumerate(self.idx):
        #     dw[word_id] += dout[idx]    
        np.add.at(dw, self.idx, dout)

        return None
    
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h: float, idx): # assume mini-batch 
        # the # of contexts is equal to # of targets
        target_W = self.embed.forward(idx)

        out = np.sum(h * target_W, axis=1)

        self.cache = (h, target_W)

        return out
    
    def backward(self, dout: np.ndarray):
        h, target_W= self.cache

        dout = dout.reshape(dout.size, 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        
        return dh

import collections

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        # sample_size: how many negative samples will be created
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter() # calculate each word's counts as special dict
        for word_id in corpus:
            counts[word_id] += 1
        
        self.vocab_size = vocab_size = len(counts) # vocab size

        self.word_p = np.zeros(vocab_size) # store word's counts from dict to ndarray
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        
        self.word_p = np.power(self.word_p, power) # such as n**0.75
        self.word_p /= np.sum(self.word_p)  # normalization so the sum of the percentages are 100%

    def get_negative_sample(self, target: np.ndarray): 
        batch_size = target.shape[0] # size of mini-batch, such as 3

        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32) # if sample_size==2, it will be a (3,2) ndarray

        for i in range(batch_size): # for each batch, set the target prob to 0, and normalization
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        
        return negative_sample

corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus, power, sample_size)
target = np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)

from layers import SigmoidWithLoss

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size) # get negative samples
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)] 
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, h, target):    
        batch_size = target.shape[0] # 1 dim
        negative_sample = self.sampler.get_negative_sample(target)

        # positive label with forward
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # negative label with forward
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)  # 1
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss]  # 2
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in
    
    def forward(self, contexts, target): # same as chapter 3 to get contexts and targets
        h = 0
        
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])  # 3
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

import pickle
from trainer import Trainer
from optimizer import Adam
from util import create_contexts_target
import ptb

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word

import pathlib


pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)