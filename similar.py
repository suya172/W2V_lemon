from gensim.models import KeyedVectors
from numpy import negative

wv = KeyedVectors.load_word2vec_format('./wikiextractor/wikiextractor/wiki.vec.pt', binary=True)
results = wv.most_similar(negative=['好き'])
for result in results:
    print(result)
