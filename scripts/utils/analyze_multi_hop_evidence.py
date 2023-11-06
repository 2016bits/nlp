import json
import argparse

def get_hop(evidence):
    max_len = 0
    for evi in evidence:
        docs = [doc for doc, _ in evi]
        docs = list(set(docs))
        if len(docs) > max_len:
            max_len = len(docs)
    return max_len

def main(args):
    in_path = args.data_path + args.dataset_name + "/processed/" + args.mode + ".json"
    out_path = args.output_path + args.dataset_name + "_multi_hop.json"

    with open(in_path, 'r') as f:
        dataset = json.load(f)
    
    results = []
    hop_dict = {}
    for inst in dataset:
        evidence = inst['evidence']
        hop = get_hop(evidence)
        if hop in hop_dict:
            hop_dict[hop] += 1
        else:
            hop_dict[hop] = 1
        results.append({
            'id': inst['id'],
            'claim': inst['claim'],
            'label': inst['label'],
            'evidence': inst['evidence'],
            'hop': hop
        })
    
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))
    
    sorted_hop_dict = sorted(hop_dict.items())
    for key, value in sorted_hop_dict:
        print(key, value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
     # data arguments
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--output_path', type=str, default='./results/analyze/')
    parser.add_argument('--dataset_name', type=str, default="FEVER")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev'])

    args = parser.parse_args()
    main(args)
