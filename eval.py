import json
import argparse
import sys
from rouge import Rouge
import numpy as np
import jieba

def rouge_score(data):
    """
    compute rouge score
    Args:
        data (list of dict including reference and candidate):
    Returns:
            res (dict of list of scores): rouge score
    """
    rouge_name = ["rouge-1", "rouge-2", "rouge-l"]
    item_name = ["f", "p", "r"]

    res = {}
    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s"%(name1, name2)] = []
    for tmp_data in data:
        origin_candidate = tmp_data['candidate']
        origin_reference = tmp_data['reference']
        assert isinstance(origin_candidate, str)
        if not isinstance(origin_reference, list):
            origin_reference = [origin_reference]

        tmp_res = []
        for r in origin_reference:
            tmp_res.append(Rouge().get_scores(refs=r, hyps=origin_candidate)[0])

        for name1 in rouge_name:
            for name2 in item_name:
                res["%s-%s"%(name1, name2)].append(max([tr[name1][name2] for tr in tmp_res]))

    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s"%(name1, name2)] = np.mean(res["%s-%s"%(name1, name2)])
    return res

def load_file(filename):
    data = []
    with open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
        f.close()
    return data

def proline(line):
    return " ".join([w for w in jieba.cut("".join(line.strip().split()))])


def compute(golden_file, pred_file, return_dict=True):
    golden_data = load_file(golden_file)
    pred_data = load_file(pred_file)

    if len(golden_data) != len(pred_data):
        raise RuntimeError("Wrong Predictions")

    eval_data = [{"reference": [proline(gg) for gg in g["tgt"]], "candidate": proline(p["tgt"][0]) if isinstance(p["tgt"], list) else proline(p["tgt"])} for g, p in zip(golden_data, pred_data)]
    return rouge_score(eval_data)

def main():
    argv = sys.argv
    print("预测结果：{}, 测试集: {}".format(argv[1], argv[2]))
    print(compute(argv[2], argv[1]))


if __name__ == '__main__':
    main()
