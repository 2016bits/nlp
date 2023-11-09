import json
import csv
import os
from tqdm import tqdm


def convert(count, in_path, out_path):
    # 打开 JSON Lines 文件以读取数据
    with open(in_path, "r") as jsonl_file:
        # 读取 JSON Lines 文件的每一行
        lines = jsonl_file.readlines()

    # 打开 CSV 文件以写入数据
    with open(out_path, "w", newline="") as csv_file:
        # 创建 CSV 写入器
        csv_writer = csv.writer(csv_file)

        # 写入 CSV 文件的标题行
        csv_writer.writerow(["id", "text", "title"])

        # 遍历 JSON Lines 文件的每一行
        for line in lines:
            # 解析 JSON 行
            data = json.loads(line)

            # 提取数据并写入 CSV 文件
            csv_writer.writerow([count, data["text"], data["id"]])
            count += 1
    return count

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
    in_dir = './wiki-pages'
    out_dir = './outputs/'
    files = [f for f in iter_files(in_dir)]
    count = 0
    for file in tqdm(files):
        filename = file.split('/')[-1]
        in_path = file
        out_path = (out_dir + filename).replace('jsonl', 'csv')
        count = convert(count, in_path, out_path)