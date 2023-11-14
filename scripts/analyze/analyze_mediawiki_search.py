import json
import argparse

def main(args):
    in_path = args.data_path + args.dataset_name + "_" + args.mode + "_mediawiki_search_claim_3.jsonl"
    out_path = args.output_path + args.dataset_name + "_" + args.mode + "_multi_hop.json"

    dataset = []
    with open(in_path, "r") as f:
        for line in f.readlines():
            dataset.append(json.loads(line.strip()))
    
    noun_num_dict = {}
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
    
    sorted_hop_dict = sorted(hop_dict.items())
    for key, value in sorted_hop_dict:
        print(key, value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
     # data arguments
    parser.add_argument('--data_path', type=str, default='./results/search/')
    parser.add_argument('--output_path', type=str, default='./results/analyze/')
    parser.add_argument('--dataset_name', type=str, default="FEVER")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev'])

    args = parser.parse_args()
    main(args)
