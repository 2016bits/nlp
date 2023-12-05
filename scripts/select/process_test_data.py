import json
import argparse
import sqlite3
from tqdm import tqdm

class WikiPage:
    def __init__(self, db_path, db_table):
        self.db_table = db_table
        conn = sqlite3.connect(db_path)
        self.c = conn.cursor()
    
    def search_wiki(self, titles):
        # 返回标题为titles的所有页面
        sql = """select * from {} where id in ({seq})""".format(self.db_table, seq=','.join(['?'] * len(titles)))
        cursor = self.c.execute(sql, titles)
        return cursor
    
    def get_pages(self, titles):
        # 根据titles得到所有句子的列表，列表元素为[title, line_id, line_text]
        cursor = self.search_wiki(titles)
        results = []
        for row in cursor:
            title = row[0]
            lines = row[2]
            evidence = []
            line_list = lines.split('\n')
            for line in line_list:
                if line and line[0].isdigit:
                    line_chunk = line.split('\t')
                    if len(line_chunk) > 1 and len(line_chunk[1]) > 0:
                        line_id = eval(line_chunk[0])
                        line_text = line_chunk[1]
                        evidence.append([title, line_id, line_text])
            results.append(evidence)
        return results

def main(args):
    wiki = WikiPage(args.db_path, args.db_table)
    with open(args.in_path, 'r') as f:
        dataset = json.load(f)
    
    results = []
    for data in tqdm(dataset):
        id = data['id']
        claim = data['claim']
        label = data['gold_label']
        gold_evidence = data['gold_evidence']
        pred_evidence = wiki.get_pages(data['parsed_title'])
        results.append({
            'id': id,
            'claim': claim,
            'label': label,
            'gold_evidence': gold_evidence,
            'pred_evidence': pred_evidence
        })
    
    with open(args.out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))
    print("finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default="./results/parse/FEVER_test_merge_parse_search_keyword_results.json")
    parser.add_argument('--out_path', type=str, default="./results/select/test_data.json")
    parser.add_argument('--db_path', type=str, default="./data/Wikipedia/data/wikipedia.db")
    parser.add_argument('--db_table', type=str, default="documents")

    args = parser.parse_args()
    main(args)
