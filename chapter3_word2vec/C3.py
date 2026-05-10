from create_context_target_better import create_contexts_target

from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from layers import MatMul
from util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

def convert_one_hot(corpus: np.ndarray, vocab_size: int):

    shape0 = corpus.shape[0]
    if corpus.ndim not in (1,2):
        raise ValueError("should be 2d or 1d array")
    if corpus.ndim == 1:
        one_hot_matrix = np.zeros((shape0, vocab_size), dtype=np.int32)

        for idx, word_id in enumerate(corpus): 
            one_hot_matrix[idx, word_id] = 1

    elif corpus.ndim == 2:
        shape1 = corpus.shape[1]
        one_hot_matrix = np.zeros((shape0, shape1, vocab_size), dtype=np.int32)

        for row_idx, row in enumerate(corpus):
            for idx, word_id in enumerate(row): 
                one_hot_matrix[row_idx, idx, word_id] = 1

    return one_hot_matrix

contexts, targets = create_contexts_target(corpus, window_size=1)

contexts = convert_one_hot(contexts, vocab_size)
targets = convert_one_hot(corpus=targets, vocab_size=vocab_size)

def softmax(x: np.ndarray):
    if x.ndim == 1:
        x = x - np.max(x)
        numerator = np.exp(x)
        denominator = np.sum(numerator)
        result = numerator / denominator
    elif x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        numerator = np.exp(x)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        result = numerator / denominator
    return result

def cross_entropy_error(y_predict: np.ndarray, y: np.ndarray, eps=1e-8):
    # If y_predict is 1D, it means there is only one sample.
    # Reshape it to (1, class_num), so the following batch logic can be reused.

    if y_predict.ndim == 1: # 1 sample
        y = y.reshape(1, y.size)
        y_predict = y_predict.reshape(1, y_predict.size)

    if y.size == y_predict.size:
        y = y.argmax(axis=1)

    batch_size = y_predict.shape[0]

    return -np.sum(np.log(y_predict[np.arange(batch_size), y] + eps)) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y_predict = None
        self.y = None
        self.loss = None

    def forward(self, x: np.ndarray, y: np.ndarray):
        self.y = y
        self.y_predict = softmax(x)

        if self.y.size == self.y_predict.size:
            self.y = self.y.argmax(axis=1)
            
        self.loss = cross_entropy_error(self.y_predict, self.y) # it will check whether y is one-hot or label and works for both

        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.y_predict.shape[0]

        dx = self.y_predict.copy()
        dx[np.arange(batch_size), self.y] -= 1

        dx = dx / batch_size
        return dx * dout

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        context_words_before = contexts[:, 0]
        context_words_after = contexts[:, 1]
        d0 = self.in_layer0.forward(context_words_before)
        d1 = self.in_layer1.forward(context_words_after)
        d_total = (d0 + d1) / 2
        score = self.out_layer.forward(d_total)
        self.loss = self.loss_layer.forward(score, target)

        return self.loss
    
    def backward(self, dout=1):
        dx_loss_layer = self.loss_layer.backward(dout)
        dx_out_layer = self.out_layer.backward(dx_loss_layer)
        dx_div_by_2 =  dx_out_layer / 2
        dx_in_layer0 = self.in_layer0.backward(dx_div_by_2)
        dx_in_layer1 = self.in_layer1.backward(dx_div_by_2)

        return None

# train

from trainer import Trainer
from optimizer import Adam
from util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])