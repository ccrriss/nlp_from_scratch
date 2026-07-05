import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from optimizer import SGD
from trainer import RnnlmTrainer
from util import eval_perplexity
import ptb
from C6 import RnnLM

batch_size = 20 # N = 20
wordvec_size = 100 # D = 100
hidden_size = 100 # H = 100
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id) # V
xs = corpus[:-1]
ts = corpus[1:]

model = RnnLM(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# evaluate with test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print(f"test perplxity: {ppl_test}")

model.save_params()