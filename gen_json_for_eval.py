import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--format_path', type=str, default='./data/valid.jsonl')
parser.add_argument('--input_path', type=str, default='./data/gen_copy_valid.txt')
parser.add_argument('--output_path', type=str, default='./data/gen_copy_valid.jsonl')
args = parser.parse_args()

f1 = open(args.format_path, "r", encoding='utf-8')
f2 = open(args.input_path, "r", encoding='utf-8')
f3 = open(args.output_path, "w", encoding='utf-8')
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