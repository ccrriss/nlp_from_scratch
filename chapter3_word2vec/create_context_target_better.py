from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from layers import MatMul
from util import preprocess

# text = 'You say goodbye and I say hello.'
# corpus, word_to_id, id_to_word = preprocess(text)

def create_contexts_target(corpus, window_size=1):
    targets = corpus[window_size: -window_size]
    contexts = []
    corpus_size = len(corpus)

    for center_idx in range(window_size, corpus_size-window_size): 
        cs = []
        for offset in range(-window_size, window_size + 1):        
            if offset == 0:
                continue
            cs.append(corpus[center_idx + offset])
        contexts.append(cs)
    return np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)

# contexts, targets = create_contexts_target(corpus, window_size=1)
# print(contexts, targets)

        
