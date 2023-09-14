import json
import argparse

def process_evidence(evidences):
    evidence_list = []
    for evidence_group in evidences:
        for evidence in evidence_group:
            evi = [evidence[2], evidence[3]]
            if evi not in evidence_list:
                evidence_list.append(evi)
    return evidence_list

def main(args):
    data_list = []
    with open(args.in_file, 'r') as f1:
        for line in f1:
            data = json.loads(line)
            if data['label'] == "NOT ENOUGH INFO":
                data_list.append({
                    'id': data['id'],
                    'claim': data['claim'],
                    'label': data['label'],
                    'evidence': []
                })
            else:
                evidence = process_evidence(data['evidence'])
                data_list.append({
                    'id': data['id'],
                    'claim': data['claim'],
                    'label': data['label'],
                    'evidence': evidence
                })
    with open(args.out_file, 'w') as f2:
        json.dump(data_list, f2, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='/data/yangjun/LLM/wikillm/data/FEVER/raw/paper_test.jsonl')
    parser.add_argument('--out_file', type=str, default='/data/yangjun/LLM/wikillm/data/FEVER/processed/test2.json')

    args = parser.parse_args()
    main(args)
