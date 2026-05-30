import pickle

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

pkl_file = BASE_DIR / 'cbow_params.pkl'

from util import most_similar


with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']

for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

from util import analogy

print(analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs, top=5))
print(analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs))
print(analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs))
print(analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs))