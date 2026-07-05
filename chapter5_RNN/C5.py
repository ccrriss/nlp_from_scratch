import numpy as np
from layers import Embedding, Affine, SoftmaxWithLoss

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        Wx, Wh, b = self.params

        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next: np.ndarray): 
        x, h_prev, h_next = self.cache
        Wx, Wh, b = self.params

        dt = dh_next * (1 - np.power(h_next, 2))

        db = np.sum(dt, axis=0)
        dx = np.dot(dt, Wx.T)
        dWx = np.dot(x.T, dt)

        dh_prev = np.dot(dt, Wh.T)
        dWh = np.dot(h_prev.T, dt)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev
    
class TimeRNN:
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h
    
    def reset_state(self):
        self.h = None

    def forward(self, xs: np.ndarray): 
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape
        
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        
        for t in range(T):
            layer = RNN(Wx, Wh, b)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs

    def backward(self, dhs: np.ndarray): # dhs.shape is (N, T, H)
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.zeros((N, T, D), dtype='f')
        dh = 0
        grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        for t in reversed(range(T)):
            dh = dh + dhs[:, t, :] # to get the sum then go to backward
            layer = self.layers[t]
            dx, dh = layer.backward(dh) # the result of backward
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads): # Wx, Wh, b for each layer
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad 

        self.dh = dh

        return dxs

class TimeEmbedding:
    def __init__(self, W: np.ndarray):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
    
    def forward(self, xs: np.ndarray):
        N, T = xs.shape
        V, D = self.W.shape
        W = self.W

        out = np.empty((N, T, D), dtype="f")
        self.layers = [] 

        for t in range(T):
            layer = Embedding(W)
            out[:, t, :] = layer.forward(xs[:, t]) 
            self.layers.append(layer)
 
        return out
    
    def backward(self, dout: np.ndarray): # N, T, D as shape 
        W, = self.params
        N, T, D = dout.shape

        grad = np.zeros_like(W)

        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None
    
class TimeAffine:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.hs = None
    
    def forward(self, hs: np.ndarray): # N, T, H as the shape
        N, T, H = hs.shape
        W, b = self.params # W shape: (H, V)  ; b shape: (V,)
        H_W, V = W.shape

        assert H == H_W

        rhs = hs.reshape(N*T, -1) # (N*T, H)
        out = np.dot(rhs, W) + b  # (N*T, V)
        self.hs = hs

        return out.reshape(N, T, -1)

    def backward(self, dout: np.ndarray): # dout's shape is N, T, V
        hs = self.hs
        N, T, H = hs.shape
        W, b = self.params # W:(H, V)

        rhs = hs.reshape(N*T, -1)     # (N*T, H)
        dout = dout.reshape(N*T, -1) # (N*T, V)

        db = np.sum(dout, axis=0)
        dhs = np.dot(dout, W.T) 
        dW = np.dot(rhs.T, dout) 

        dhs = dhs.reshape(*hs.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dhs

from functions import softmax

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1
    
    def forward(self, xs: np.ndarray, ys: np.ndarray): # (N, T, V) as x.shape, t.shape shouldbe (N,T)
        N, T, V = xs.shape

        if ys.ndim == 3:  # change one-hot to label
            ys = ys.argmax(axis=2)

        mask = (ys != self.ignore_label) 

        xs = xs.reshape(N * T, V)
        ys = ys.reshape(N * T)
        mask = mask.reshape(N * T)

        y_predicts = softmax(xs) # shape is (N*T, V)
        ls = np.log(y_predicts[np.arange(N * T), ys])  
        ls *= mask 
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ys, y_predicts, mask, (N, T, V))
        return loss
    
    def backward(self, dout=1): # (N,)
        ys, y_predicts, mask, (N, T, V) = self.cache

        dx = y_predicts # (N * T, V)
        dx[np.arange(N * T), ys] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis] # (N*T, 1) for broadcasting

        dx = dx.reshape((N, T, V))

        return dx

class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # initialize weights 
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # layers
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs: np.ndarray, ts: np.ndarray):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.rnn_layer.reset_state()