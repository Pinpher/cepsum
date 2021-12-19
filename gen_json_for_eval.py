import json

f1 = open('./data/valid.jsonl', "r", encoding='utf-8')
f2 = open('./data/gen_copy_valid.txt', "r", encoding='utf-8')
f3 = open('./data/gen_copy_valid.jsonl', "w", encoding='utf-8')
f1_lines = f1.readlines()
f2_lines = f2.readlines()

for i, line in enumerate(f1_lines):
    #print(line)
    j = json.loads(line)
    j['tgt'] = [f2_lines[i][:-1]]
    f3.write(json.dumps(j, ensure_ascii=False) + "\n")
    #break

f1.close()
f2.close()
f3.close()