from gensim.models import KeyedVectors
import numpy as np
import spacy
import re
class Color:
	BLACK          = '\033[30m'#(文字)黒
	RED            = '\033[31m'#(文字)赤
	GREEN          = '\033[32m'#(文字)緑
	YELLOW         = '\033[33m'#(文字)黄
	BLUE           = '\033[34m'#(文字)青
	MAGENTA        = '\033[35m'#(文字)マゼンタ
	CYAN           = '\033[36m'#(文字)シアン
	WHITE          = '\033[37m'#(文字)白
	COLOR_DEFAULT  = '\033[39m'#文字色をデフォルトに戻す
	BOLD           = '\033[1m'#太字
	UNDERLINE      = '\033[4m'#下線
	INVISIBLE      = '\033[08m'#不可視
	REVERCE        = '\033[07m'#文字色と背景色を反転
	BG_BLACK       = '\033[40m'#(背景)黒
	BG_RED         = '\033[41m'#(背景)赤
	BG_GREEN       = '\033[42m'#(背景)緑
	BG_YELLOW      = '\033[43m'#(背景)黄
	BG_BLUE        = '\033[44m'#(背景)青
	BG_MAGENTA     = '\033[45m'#(背景)マゼンタ
	BG_CYAN        = '\033[46m'#(背景)シアン
	BG_WHITE       = '\033[47m'#(背景)白
	BG_DEFAULT     = '\033[49m'#背景色をデフォルトに戻す
	RESET          = '\033[0m'#全てリセット
	
# テキストのベクトルを計算
def text_get_vector(text):
    sum_vec = np.zeros(200)
    word_count = 0
    _doc = nlp(text)
    for sent in _doc.sents:
        for token in sent:
            to_consider = ['NOUN','VERB','ADV','ADJ','PRON','PROPN','INTJ'] # 品詞によってはvectorを加算する必要がない
            if token.pos_ in to_consider:
                try:
                    v = wv[token.text]
                except KeyError:
                    failed_list.append(token.text)
                    v = np.zeros_like(wv['夢'])
                sum_vec = sum_vec + v # 単語のvectorを足し合わせる
                word_count = word_count + 1 # 加算した単語数をカウント
    return sum_vec / word_count # 平均値を求める

# 単語リスト内の単語のvectorの平均値を算出
def word_list_get_tendency_vector(word_list):
     assert len(word_list) > 0, 'wordlist is empty' # wordlistのサイズが0の場合エラーを吐く
     sum_vec = np.zeros(200)
     word_count = len(word_list)
     for word in word_list:
          sum_vec = sum_vec + text_get_vector(word)
     return sum_vec / word_count

# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

wv = KeyedVectors.load_word2vec_format('./entity_vector.model.bin', binary=True)
nlp = spacy.load('ja_ginza')
lemon_text = open('./Lemon.txt','r', encoding='utf8').read()
lemon_text = re.sub(r'[\s「」]','',lemon_text)
lemon_noun_chunks_list = []
failed_list = []
doc = nlp(lemon_text)

#region 
# # nodeをlist化
# while lemon_node:
#     fields = lemon_node.feature.split(',')
#     while len(fields) < 6:
#          fields.append('*')
#     try:
#         lemon_list.append((lemon_node.surface,fields[0],fields[1],fields[2],fields[3],fields[4],fields[5]))
#     except:
#         print(f'{Color.RED}{lemon_node.surface},{fields}{Color.RESET}')
#         exit()
#     lemon_node = lemon_node.next

# # listを後ろから見て行って、条件に沿って品詞を結合
# compounding = False
# to_continue = ['名詞','動詞','形容詞','形容動詞','連体詞','助動詞','接尾辞','代名詞']
# for i in reversed(range(len(lemon_list))):
#     fields = lemon_list[i]
#     if fields[1] == '名詞' or fields[1] == '動詞' or fields[1] == '形容詞' or fields[1] == '形容動詞':
#         d = 1
#         try:
#             vec = wv[fields[1]]
#         except:
#             failed_list.append(fields[1])
#             vec = np.zeros_like(wv['夢'])
#     else:
#         d = 0

#     if compounding:
#         if fields[1] in to_continue or (fields[1] == '助詞' and fields[2] == '接続助詞'):
#             length = len(lemon_compoundnouns_list)
#             lemon_compoundnouns_list[length-1][0] = fields[0] + lemon_compoundnouns_list[length-1][0]

#             if d == 1:
#                 lemon_compoundnouns_list[length-1][1] = lemon_compoundnouns_list[length-1][1] + vec

#             lemon_compoundnouns_list[length-1][2] = lemon_compoundnouns_list[length-1][2] + d
#         else:
#             compounding = False
#     else:
#         if fields[1] == '名詞' or (fields[1] == '助詞' and fields[2] == '準体助詞') or fields[1] == '接尾辞':
#             lemon_compoundnouns_list.append([fields[0],vec,1]) 
#             compounding = True

#     # Debug        
#     # for compoundnouns in lemon_compoundnouns_list:
#     #     print (f'{Color.BLUE}{[compoundnouns[0], compoundnouns[2]]}{Color.RESET}')
#     # print('----------------------------------------------------------------')

# # 順番を戻しておく
# lemon_compoundnouns_list.reverse()

# # ベクトルの平均値を求める
# for compoundnouns in lemon_compoundnouns_list:
#     compoundnouns[1] = compoundnouns[1] / compoundnouns[2]


# for compoundnouns in lemon_compoundnouns_list:
#     print(compoundnouns[0])

# print(f'{Color.RED}failed:{failed_list}{Color.RESET}')
#endregion

# 重複なしで名詞句を抽出
_s = set({})
for chunk in doc.noun_chunks:
    _s.add(chunk.text)
for chunk_t in _s:
     lemon_noun_chunks_list.append((chunk_t, text_get_vector(chunk_t)))
del _s

old_list = ['美しい詩','洒落た切子細工','オードコロン','オードキニン','典雅なロココ趣味の浮模様を持った琥珀色や翡翠色の香水壜','煙管','小刀','石鹼','煙草']
new_list = ['さまざまな縞模様を持った花火の束','中山寺の星下り','花合戦','枯れすすき','鼠花火','壊れかかった街','裏通り','植物']
old_v = word_list_get_tendency_vector(old_list)
new_v = word_list_get_tendency_vector(new_list)

for _ in lemon_noun_chunks_list:
     word = _[0]
     v = _[1]
     old_similar = cos_sim(v,old_v)
     new_similar = cos_sim(v,new_v)
     if old_similar > new_similar:
          print(f'{word} {Color.GREEN}{old_similar}{Color.RESET},{new_similar} -> old')
     if old_similar < new_similar:
          print(f'{word} {old_similar},{Color.GREEN}{new_similar}{Color.RESET} -> new')
     if old_similar == new_similar:
          print(f'{word} {Color.BLUE}{old_similar}{Color.RESET},{Color.BLUE}{new_similar}{Color.RESET} -> draw')

print(f'{Color.RED}failed:{failed_list}{Color.RESET}')
          