import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
param_path = BASE_DIR / 'chapter8_Attention'
import pickle

import numpy as np
from layers import Softmax
from chapter7_generate_texts.C7 import Encoder, Seq2seq
from time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
    
    def forward(self, hs: np.ndarray, a: np.ndarray): # hs.shape (N, T, H) a.shape (N, T)
        N, T, H = hs.shape
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = ar * hs  # (N, T, H)
        c = np.sum(t, axis=1) # (N, H)

        self.cache = N, T, H, ar, hs
        return c
    
    def backward(self, dout: np.ndarray): # (N, H)
        N, T, H, ar, hs= self.cache

        dt = dout.reshape(N, 1, H).repeat(T, axis=1)
        dar = hs * dt
        dhs = ar * dt
        da = np.sum(dar, axis=2)

        return dhs, da

class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None
    
    def forward(self, hs: np.ndarray, h: np.ndarray) -> np.ndarray: # h: (N,H) hs: (N, T, H)
        N, T, H = hs.shape
        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hr * hs # (N, T, H)
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s) # (N, T)

        self.cache = hs, hr

        return a
    
    def backward(self, dout: np.ndarray): # (N,T)
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(dout) # (N,T)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhr = hs * dt
        dhs = hr * dt
        dh = np.sum(dhr, axis=1)

        return dhs, dh

class Attention:
    def __init__(self):
        self.attweight = AttentionWeight()
        self.weightsum = WeightSum()

        self.params, self.grads = [], []

        self.attention_weight = None

    def forward(self, hs:np.ndarray, h:np.ndarray):
        a = self.attweight.forward(hs, h)
        c = self.weightsum.forward(hs, a)

        self.attention_weight = a

        return c

    def backward(self, dout):
        dhs0, da = self.weightsum.backward(dout)
        dhs1, dh = self.attweight.backward(da)

        dhs = dhs0 + dhs1
        
        return dhs, dh

class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None
        self.cache = None
    
    def forward(self, hs_enc: np.ndarray, hs_dec: np.ndarray): # hs_encoder and hs_decoder
        N, T_enc, H = hs_enc.shape
        N, T_dec, H = hs_dec.shape

        cs = np.zeros((N, T_dec, H), dtype='f')
        self.layers = []
        self.attention_weights = []

        for t in range(T_dec):
            att_layer = Attention()
            c = att_layer.forward(hs_enc, hs_dec[:, t, :]) # (N, H)
            cs[:, t, :] = c

            self.layers.append(att_layer)
            self.attention_weights.append(att_layer.attention_weight)

        self.cache = (T_enc,)
        return cs

    def backward(self, dout: np.ndarray):  # N, T_dec, H
        N, T_dec, H = dout.shape
        T_enc, = self.cache

        dhs_enc_total = np.zeros((N, T_enc, H), dtype=np.float32)
        dhs_dec = np.zeros((N, T_dec, H), dtype=np.float32)

        for t in reversed(range(T_dec)):
            att_layer = self.layers[t]
            dhs_enc, dh_dec = att_layer.backward(dout[:, t, :])
            dhs_enc_total += dhs_enc
            dhs_dec[:, t, :] = dh_dec

        return dhs_enc_total, dhs_dec

class AttentionEncoder(Encoder):
    def forward(self, xs: np.ndarray) -> np.ndarray:
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)

        return hs

    def backward(self, dhs: np.ndarray):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)

        return dout

class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H+H, V) / np.sqrt(H+H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)

        self.layers = [self.embed, self.lstm, self.attention, self.affine]
        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    def forward(self, xs: np.ndarray, hs_enc: np.ndarray):
        xs = self.embed.forward(xs)

        self.lstm.set_state(hs_enc[:, -1, :]) # set h
        hs_dec = self.lstm.forward(xs) # (N, T_dec, H)
        cs = self.attention.forward(hs_enc, hs_dec) # (N, T_dec, H)
        cs = np.concatenate((cs, hs_dec), axis=2) # (N, T_dec, 2H)
        score = self.affine.forward(cs) # (N, T, V)

        self.cache = hs_enc

        return score
    
    def backward(self, dscore):
        hs_enc = self.cache
        N, T_enc, H = hs_enc.shape

        dcs = self.affine.backward(dscore)
        dcs, dhs_dec1 = dcs[:, :, :H], dcs[:, :, H:]
        dhs_enc_total, dhs_dec0 = self.attention.backward(dcs)
        dhs_dec = dhs_dec1 + dhs_dec0
        dxs = self.lstm.backward(dhs_dec)
        dhs_enc_total[:, -1, :] += self.lstm.dh

        dout = self.embed.backward(dxs)

        return dhs_enc_total

    def generate(self, hs_enc: np.ndarray, start_id: int, sample_size:int):
        _, T_enc, H = hs_enc.shape
        h = hs_enc[:, -1, :]

        sampled = []
        sampled_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sampled_id).reshape((1,1))
            out = self.embed.forward(x) # (1, 1, D)
            out = self.lstm.forward(out) # (1, 1, H)

            c = self.attention.forward(hs_enc, out) # (1, 1, H)
            c = np.concatenate((c, out), axis=2) # (1,1,2H)
            score = self.affine.forward(c)
            sampled_id = np.argmax(score.flatten())
            sampled.append(sampled_id)
        
        return sampled

class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def save_params(self, file_name='attentionseq2seq.pkl'):
        file_path = param_path / file_name
        with open(file_path, 'wb') as f:
            pickle.dump(self.params, f)