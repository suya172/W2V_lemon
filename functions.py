import MeCab
import ipadic
from gensim.models import KeyedVectors
import numpy as np


mt = MeCab.Tagger('')
wv = KeyedVectors.load_word2vec_format('./entity_vector.model.bin', binary=True)

# テキストのベクトルを計算
def text_get_vector(text):
    sum_vec = np.zeros(200)
    word_count = 0
    node = mt.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        # 名詞、動詞、形容詞に限定
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            sum_vec += wv[node.surface]
            word_count += 1
        node = node.next

    return sum_vec / word_count


# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# # Mecabのnodeをlist化
# def mecab_list(node):
#     tagger.parse('')
#     word_class = []
#     while node:
#         word = node.surface
#         wclass = node.feature.split(',')
#         if wclass[0] != u'BOS/EOS':
#             if wclass[6] == None:
#                 word_class.append((word,wclass[0],wclass[1],wclass[2],""))
#             else:
#                 word_class.append((word,wclass[0],wclass[1],wclass[2],wclass[3],wclass[4],wclass[5],wclass[6]))
#         node = node.next
#     return word_class



if __name__ == "__main__":
    exit()
