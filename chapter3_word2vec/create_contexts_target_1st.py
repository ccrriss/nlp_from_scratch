from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from layers import MatMul
from util import preprocess

# text = 'You say goodbye and I say hello.'
# corpus, word_to_id, id_to_word = preprocess(text)

def create_contexts_target(corpus, window_size=1):
    corpus_size = len(corpus)
    sample_size = corpus_size - window_size * 2 # not including first and last # of window_size elements

    contexts = np.zeros((sample_size, window_size*2), dtype=np.int32) 
    targets = np.zeros(sample_size, dtype=np.int32) 
    for center_idx in range(window_size, corpus_size-window_size): 
        sample_idx = center_idx - window_size # always start from 0
        for i in range(1, window_size + 1):   # from 1 to windowsize, eg 1, 2 for window_size == 2, 1 for window_size ==1        
            left_idx = center_idx - i
            right_idx = center_idx + i
            contexts[sample_idx, i*2 - 2] = corpus[left_idx]
            contexts[sample_idx, i*2 - 1] = corpus[right_idx]
        targets[sample_idx] = corpus[center_idx]
    return contexts, targets

# contexts, targets = create_contexts_target(corpus, window_size=3)
# print(contexts, targets)

        
