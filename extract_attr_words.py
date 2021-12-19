from tqdm import tqdm

all_words = set()
general_words = set()

def nonmatch(word1, word2):
    return len(set(word1) & set(word2)) == 0

def process(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            _, _, src, tgt = line.split("\t\t")
            src = src.split()
            tgt = tgt.split()
            for tgt_word in tgt:
                all_words.add(tgt_word)
                if all([nonmatch(tgt_word, src_word) for src_word in src]):
                    general_words.add(tgt_word)

process("data/cut_train.txt")
process("data/cut_valid.txt")
attr_words = all_words - general_words - set(["[UNK]", "[CLS]", "[SEP]"])
with open("vocab/attr_words.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(list(attr_words)))



''' simple process '''
simple_attr_words = set()

def simple_process(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            _, values, _, _ = line.split("\t\t")
            for value in values.split():
                simple_attr_words.add(value)

simple_process("data/cut_train.txt")
simple_process("data/cut_train.txt")
with open("vocab/simple_attr_words.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(list(simple_attr_words)))
