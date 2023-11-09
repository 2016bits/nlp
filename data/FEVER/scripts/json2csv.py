import json
import csv
import os
from tqdm import tqdm

def get_document(evidence_list):
    """get documents from given evidence"""
    doc = []
    for evidence in evidence_list:
        doc_list = []
        for evi, _ in evidence:
            if evi not in doc:
                doc_list.append(evi)
        if doc_list not in doc:
            doc.append(doc_list)
    return doc

def convert(in_path, out_path):
    # 打开 JSON 文件以读取数据
    with open(in_path, "r") as json_file:
        # 读取 JSON 文件的每一行
        dataset = json.load(json_file)

    # 打开 CSV 文件以写入数据
    with open(out_path, "w", newline="") as csv_file:
        # 创建 CSV 写入器
        csv_writer = csv.writer(csv_file)

        # 写入 CSV 文件的标题行
        csv_writer.writerow(["claim", "doc", "label", "evidence"])

        # 遍历 JSON 数据
        for item in dataset:
            claim = item["claim"]
            label = item["label"]
            evidence = item["evidence"]
            doc = get_document(evidence)

            # 将数据写入 CSV 文件
            csv_writer.writerow([claim, doc, label, evidence])

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)

if __name__ == '__main__':
    in_path = './processed/test.json'
    out_path = './processed/test.csv'
    convert(in_path, out_path)