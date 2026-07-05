import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import numpy as np
from functions import softmax
from chapter6_gatedRNN.C6 import RnnLM, BetterRnnlm

class RnnlmGen(RnnLM):
    def generate(self, start_id, skip_ids=None, sample_size=100): # start_id & skip_id
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