import pickle
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
param_path = BASE_DIR / 'chapter7_generate_texts'

import numpy as np
from functions import softmax
from chapter6_gatedRNN.C6 import RnnLM, BetterRnnlm
from time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
from base_model import BaseModel

class RnnlmGen(RnnLM):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)
            score = self.predict(x) # (1, V) as the result shape
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), p=p)

            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        return word_ids
    
    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c
    
    def set_state(self, state):
        self.lstm_layer.set_state(*state)
    
class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), p=p)

            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        return word_ids
    
    def get_state(self):
        states = []
        for lstm_layer in self.lstm_layers:
            states.append((lstm_layer.h, lstm_layer.c))
        return states
    
    def set_state(self, states):
        for lstm_layer, state in zip(self.lstm_layers, states):
            lstm_layer.set_state(*state)

class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None  

    def forward(self, xs:np.ndarray) -> np.ndarray : # xs.shape is (N, T)
        xs = self.embed.forward(xs) # (N, T, D)
        hs = self.lstm.forward(xs) # (N, T, H)
        self.hs = hs                    
        
        return hs[:, -1, :]    # (N, H)

    def backward(self, dh): # dh 是一个时刻的, 但TimeLSTM需要的是dhs, 也就是(N, T, H)
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
       
class Decoder: 
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs: np.ndarray, h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out) # (N, T, V)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh

        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)
        
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1,1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out) # (1,V)
            
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled

class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs:np.ndarray, ts:np.ndarray) -> float:
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs) # (N, H)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)

        return loss
    
    def backward(self, dout=1):
        dscore = self.softmax.backward(dout)
        dh = self.decoder.backward(dscore)
        dout = self.encoder.backward(dh)

        return dout
    
    def generate(self, xs: np.ndarray, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled

    def save_params(self, file_name='seq2seq.pkl'):
        file_path = param_path / file_name
        with open(file_path, 'wb') as f:
            pickle.dump(self.params, f)

class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H+D, 4*H) / np.sqrt(H+D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b =  np.zeros(4*H).astype('f')
        affine_W = (rn(H+H, V) / np.sqrt(H+H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None
    
    def forward(self, xs:np.ndarray, h:np.ndarray): # xs.shape:(N, T), h.shape:(N, H)
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs) # (N, T, D)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2) # (N, T, H + D)

        out = self.lstm.forward(out) # (N, T, H)
        out = np.concatenate((hs, out), axis=2) # (N, T, H + H)

        score = self.affine.forward(out) # (N, T, V)

        self.cache = H
        return score

    def backward(self, dscore: np.ndarray): # (N, T, V)
        H = self.cache

        dout = self.affine.backward(dscore) # (N, T, H + H)
        dhs1, dhs_lstm = dout[:, :, :H], dout[:, :, H:]
        dout = self.lstm.backward(dhs_lstm) # (N, T, H + D)
        dhs0, dembed = dout[:, :, :H], dout[:, :, H:]

        dout = self.embed.backward(dembed)  # (N, T)

        dh = self.lstm.dh 
        dh_total = dh + np.sum(dhs1, axis=1) + np.sum(dhs0, axis=1) # (N, H)
        return dh_total
    
    def generate(self, h: np.ndarray, start_id: int, sample_size: int): # h.shape: (1, H)
        _, H = h.shape
        
        sampled = []
        sampled_id = start_id
        self.lstm.set_state(h)
        peeky_h = h.reshape(1, 1, H)

        for _ in range(sample_size):
            x = np.array([sampled_id]).reshape((1,1))
            out = self.embed.forward(x) # (1, 1, D)

            out = np.concatenate((peeky_h, out), axis=2)

            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)

            score = self.affine.forward(out) # (1,1,V)
            sampled_id = np.argmax(score.flatten())
            sampled.append(sampled_id)
        
        return sampled
    
class PeekySeq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        self.encoder = Encoder(V, D, H)
        self.peekyDecoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
    
        self.params = self.encoder.params + self.peekyDecoder.params
        self.grads = self.encoder.grads + self.peekyDecoder.grads
    
    def forward(self, xs: np.ndarray, ts:np.ndarray):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)
        score = self.peekyDecoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)

        return loss

    def backward(self, dout=1):
        dscore = self.softmax.backward(dout)
        dh = self.peekyDecoder.backward(dscore)
        dout = self.encoder.backward(dh)

        return dout
    
    def generate(self, xs:np.ndarray, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.peekyDecoder.generate(h, start_id, sample_size)
        return sampled

            
'''
和x的forward合并我需要一个尺寸和该forward相匹配的h
'''