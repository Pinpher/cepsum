ifile = "vocab/vocab.txt"
ofile = "vocab/small_vocab.txt"
size = 100000

f1 = open(ifile, "r", encoding="utf8")
f2 = open(ofile, "w", encoding="utf8")

for i in range(size):
    f2.write(f1.readline())