import MeCab
from gensim.models import KeyedVectors
import numpy as np
import functions as f

mt = MeCab.Tagger('')
wv = KeyedVectors.load_word2vec_format('./entity_vector.model.bin', binary=True)

a = wv['武術']-wv['技術']
b = wv['暴力']
print(a)
print(b)
print(f.cos_sim(a,b))
