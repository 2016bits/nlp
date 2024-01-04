import json
import re

def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub(" -LRB-", " (", title)
    title = re.sub("-RRB-", ")", title)
    title = re.sub("-COLON-", ":", title)
    return title

def main():
    in_path = "data/FEVER/SelectData/dev_eval.json"
    out_path = "data/FEVER/SelectData/dev_eval.jsonl"

    dataset = []
    with open(in_path, 'r') as fin:
        for line in fin:
            dataset.append(json.loads(line.strip()))
    
    with open(out_path, 'w') as fout:
        for data in dataset:
            evidence_list = []
            if data['label'] != 'NOT ENOUGH INFO':
                for evidence in data['evidence']:
                    _evidence = []
                    for evi in evidence:
                        _evidence.append([evi[0], evi[1], process_wiki_title(evi[2]), evi[3]])
                    evidence_list.append(_evidence)
            else:
                evidence_list = data['evidence']
            instance = {
                "id": data['id'],
                'verifiable': data['verifiable'],
                'label': data['label'],
                'claim': data['claim'],
                'evidence': evidence_list
            }
            fout.write(json.dumps(instance) + '\n')

    print("finished!")

if __name__ == '__main__':
    main()
