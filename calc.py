import MeCab
from gensim.models import KeyedVectors
import numpy as np
import functions as f

mt = MeCab.Tagger('')
wv = KeyedVectors.load_word2vec_format('./wikiextractor/wikiextractor/wiki.vec.pt', binary=True)

a = wv['小熊']-wv['子供']
b = wv['熊']
print(a)
print(b)
print(f.cos_sim(a,b))