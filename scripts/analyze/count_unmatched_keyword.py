import json
import argparse


replacements = {
    "_": " ",
    "-LRB-": "(",
    "-LSB-": "[",
    "-LCB-": "{",
    "-RCB-": "}",
    "-RRB-": ")",
    "-RSB-": "]",
    "-COLON-": ":",
}

def untokenize(text):
    for r in replacements:
        if r in text:
            text = text.replace(r, replacements[r])
    return text.lower()

def is_match(text, keywords):
    for keyword in keywords:
        if keyword not in text:
            return False
    return True

def main(args):
    data_path = args.data_path + args.dataset + "/processed/" + args.mode + ".json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    results = []
    count = 0
    for inst in dataset:
        if inst['label'] != 'NOT ENOUGH INFO':
            evidence_list = inst['evidence']
            claim = inst['claim'].lower()
            matched = False

            # 遍历每一组完整evidence
            for evidence in evidence_list:
                doc_list = [untokenize(doc) for doc, _ in evidence]
                if is_match(claim, doc_list):
                    matched = True
                    break
            
            if not matched:
                count += 1
                results.append(inst)
    
    print("count: {}".format(count))
    out_path = args.out_path + args.dataset + "_" + args.mode + "_count_unmatched_keyword.json"
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='FEVER')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--out_path', type=str, default='./results/analyze/')

    args = parser.parse_args()
    main(args)

