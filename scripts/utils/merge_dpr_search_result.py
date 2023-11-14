import argparse
import json
import ast

def main(args):
    in_path = args.in_path
    out_path = args.out_path + str(args.top) + ".json"

    with open(in_path, 'r') as f:
        dataset = json.load(f)
    
    # 按照claim合并
    merged_dataset = {}
    for inst in dataset:
        claim = inst['claim']
        keyword = inst['question']
        gold_evidence = ast.literal_eval(inst['answers'])
        evidence = [data['title'] for data in inst['ctxs']]
        if claim in merged_dataset:
            merged_dataset[claim]['keyword'].extend([keyword])
            merged_dataset[claim]['predicted_pages'].extend(evidence)
        else:
            merged_dataset[claim] = {
                "keyword": [keyword],
                "gold_evidence": gold_evidence,
                "predicted_pages": evidence
            }
    
    # 将字典转化为列表
    results = []
    for claim in merged_dataset:
        results.append({
            "claim": claim,
            "keyword": list(set(merged_dataset[claim]['keyword'])),
            "gold_evidence": merged_dataset[claim]['gold_evidence'],
            "predicted_pages": list(set(merged_dataset[claim]['predicted_pages']))
        })
    
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./results/search/results.json')
    parser.add_argument('--out_path', type=str, default='./results/search/FEVER_test_dpr_search_keyword_')
    parser.add_argument('--top', type=int, default=3)

    args = parser.parse_args()
    main(args)
