import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # root directory
param_path = BASE_DIR / 'chapter6_gatedRNN'
sys.path.append(str(BASE_DIR))

import numpy as np

from functions import sigmoid
from time_layers import TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss
from base_model import BaseModel
import pickle

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

class LSTM:
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray):  # 4 params matrix, shape should be (D, 4*H) (H, 4*H)
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev: np.ndarray, c_prev: np.ndarray): # x.shape should be N,D, h.shape should be N,H
        Wx, Wh, b = self.params  # b.shape is (4H,)
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b # (N, 4*H) as A.shape

        f = A[:, :H]
        g = A[:, H: 2*H]
        i = A[:, 2*H: 3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params # Wx: (D, 4H) Wh: (H, 4H) b: (4H,)
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache # x:(N,D) h_prev: (N, H) c_prev: (N, H)

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + o * (1 - tanh_c_next**2) * dh_next # needs understanding

        dc_prev= f * ds

        di = g * ds
        df = c_prev * ds
        do = tanh_c_next * dh_next
        dg = i * ds

        di = di * i * (1 - i) # sigmoid's back propogation
        df = df * f * (1 - f)
        do = do * o * (1 - o)
        dg = dg * (1 - g**2)

        dA = np.hstack((df, dg, di, do)) # (N, 4H)

        dx = np.dot(dA, Wx.T) # *N,D  D, 4H -> N, 4H
        dh_prev = np.dot(dA, Wh.T) # N,H H,4H -> N, 4H
        dWx = np.dot(x.T, dA)
        dWh = np.dot(h_prev.T, dA)
        db = np.sum(dA, axis=0)  

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev
    
class TimeLSTM:
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.c = None

        self.stateful = stateful

    def set_state(self, h, c=None):
        self.h = h
        self.c = c

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, xs: np.ndarray):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, H_mul_4 = Wh.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            self.h, self.c  = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs
            
    def backward(self, dhs: np.ndarray): # (N, T, H)
        Wx, Wh, b = self.params
        N, T, H = dhs.shape # same as dcs.shape
        D, H_mul_4 = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = np.zeros((N, H), dtype='f'), np.zeros((N, H), dtype='f')
        grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        for t in reversed(range(T)):
            layer = self.layers[t]
            dh = dh + dhs[:, t, :]
            dx, dh, dc = layer.backward(dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh

        return dxs
    
class RnnLM(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs: np.ndarray):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs:np.ndarray, ts:np.ndarray):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()

    def save_params(self, file_name='Rnnlm.pkl'):
        file_path = param_path / file_name
        with open(file_path, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_name='Rnnlm.pkl'):
        file_path = param_path / file_name
        with open(file_path, 'rb') as f:
            self.params = pickle.load(f)

class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs: np.ndarray):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio # xs.shape (N, T, H) (after LSTM)
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs
    
    def backward(self, dout):
        return dout * self.mask

class BetterRnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        assert wordvec_size == hidden_size, "weight tying, wordvec_size should be equal to hidden_size"

        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4*H, dtype='f')
        lstm_Wx2 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4*H, dtype='f')
        affine_b = np.zeros(V, dtype='f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs: np.ndarray, train_flg=False): # (N, T) as xs.shape
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs: np.ndarray, ts: np.ndarray, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)

        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

    def save_params(self, file_name='Rnnlm.pkl'):
        file_path = param_path / file_name
        with open(file_path, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_name='Rnnlm.pkl'):
        file_path = param_path / file_name
        with open(file_path, 'rb') as f:
            self.params = pickle.load(f)