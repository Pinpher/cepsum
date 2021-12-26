import json
import jieba

def cut(ipath, opath):
    ifile = open(ipath, "r", encoding="utf-8")
    ilines = ifile.readlines()
    ofile = open(opath, "w", encoding="utf-8")
    odict = {}
    for line in ilines:
        idict = json.loads(line)
        keys, values = [], []
        for key in idict['table']:
            keys.append(" ".join(list(jieba.cut(key))))
            values.append(" ".join(list(jieba.cut(idict['table'][key]))))
        odict['key'] = "\t".join(keys)
        odict['value'] = "\t".join(values)
        odict['src'] = " ".join(list(jieba.cut(idict['src'])))
        for t in idict['tgt']:
            odict['tgt'] = " ".join(list(jieba.cut(t)))
            out = odict["key"] + "\t\t" + odict["value"] + "\t\t[CLS] " + odict["src"] + " [SEP]\t\t[CLS] " + odict["tgt"] + " [SEP]"
            ofile.write(out)
            #ofile.write(json.dumps(odict).encode('utf-8').decode('unicode_escape'))
            ofile.write('\n')   
    ifile.close()
    ofile.close()
    return

def main():
    input_files = ["../data/train.jsonl", "../data/val.jsonl", "../data/test_public.jsonl"]
    output_files = ["../data/cut_train.txt", "../data/cut_val.txt", "../data/cut_test.txt"]
    for i in range(len(input_files)):
        cut(input_files[i], output_files[i])

if __name__ == "__main__":
    main()