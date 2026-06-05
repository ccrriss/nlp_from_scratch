import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from optimizer import SGD
import ptb
from C5 import SimpleRnnlm
from trainer import RnnlmTrainer

batch_size = 10 # N=10
wordvec_size = 100 # D=100
hidden_size = 100 # H=100
time_size = 5 # T=5
lr = 0.1
max_epoch = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size] # 0 -999, 1000 words
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1] # exclude the last one
ts = corpus[1:] # exclude the first one

model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
rnnlmTrainer = RnnlmTrainer(model, optimizer)
rnnlmTrainer.fit(xs, ts, max_epoch, batch_size, time_size)
rnnlmTrainer.plot()