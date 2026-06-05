import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from optimizer import SGD
import ptb
from C5 import SimpleRnnlm

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
data_size = len(xs) # 999
print(f"corpus size: {corpus_size}, vocabulary size: {vocab_size}")

max_iters = data_size // (batch_size * time_size) # 19, batch_size:10, time_size: 5
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# get start position of minibatch
jump = (corpus_size - 1) // batch_size # 999 // 10 = 99
offsets = [i * jump for i in range(batch_size)] # [0, 99, 198, 297, 396, 495, 594, 693, 792, 891]

for epoch in range(max_epoch): # 100, 0 to 99
    for iter in range(max_iters): # 19, 0 to 18

        # get mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i') # (10, 5)
        batch_t = np.empty((batch_size, time_size), dtype='i') # (10, 5)
        for t in range(time_size): # 5 iterations
            for i, offset in enumerate(offsets): # 10, (0,0), (1,99), (2, 198), (3, 297) ...(9. 891)
                batch_x[i ,t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
    
    # preplexity of each epoch
    ppl = np.exp(total_loss / loss_count)
    print(f"| epoch {epoch+1} | perplexity {ppl:.2f}")
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 绘制图形
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
