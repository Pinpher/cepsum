from gensim.models.word2vec import Word2Vec
import numpy as np

def convert(list_of_floats):
    return " ".join([("%.6f" % f) for f in list_of_floats])

lines = []
for line in open("data/cut_train.txt", "r", encoding="utf-8"):
    lines.append(line.strip().split())
for line in open("data/cut_valid.txt", "r", encoding="utf-8"):
    lines.append(line.strip().split())

w2v = Word2Vec(lines, vector_size=300, min_count=1)

vec = np.mean([w2v.wv[key] for key in w2v.wv.key_to_index.keys()], axis=1)

with open("vocab/vocab.txt", "w", encoding="utf8") as f:
    f.write("[UNK]" + " " + convert(vec) + "\n")
    for key in w2v.wv.key_to_index.keys():
        f.write(key + " " + convert(w2v.wv[key]) + "\n")
