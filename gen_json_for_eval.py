import json

f1 = open('./data/test_public.jsonl', "r", encoding='utf-8')
f2 = open('./data/generate_tgt.txt', "r", encoding='utf-8')
f3 = open('./data/generated.jsonl', "w", encoding='utf-8')
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