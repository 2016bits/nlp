import json
import argparse

def get_sentence_num(evidence):
    max_len = 0
    min_len = 1000
    for evi in evidence:
        if len(evi) > max_len:
            max_len = len(evi)
        if len(evi) < min_len:
            min_len = len(evi)
    return max_len, min_len

def main(args):
    in_path = args.data_path + args.dataset_name + "/processed/" + args.mode + ".json"
    out_path = args.output_path + args.dataset_name + "_" + args.mode + "_evidence_sentence_num.json"

    with open(in_path, 'r') as f:
        dataset = json.load(f)
    
    results = []
    max_sent_num_dict = {}
    min_sent_num_dict = {}

    for inst in dataset:
        evidence = inst['evidence']
        max_len, min_len = get_sentence_num(evidence)
        if max_len in max_sent_num_dict:
            max_sent_num_dict[max_len] += 1
        else:
            max_sent_num_dict[max_len] = 1
        if min_len in min_sent_num_dict:
            min_sent_num_dict[min_len] += 1
        else:
            min_sent_num_dict[min_len] = 1

        results.append({
            'id': inst['id'],
            'claim': inst['claim'],
            'label': inst['label'],
            'evidence': inst['evidence'],
            'max_sent_num': max_len,
            'min_sent_num': min_len
        })
    
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

    print("max_sent_num:")
    sorted_max_num_dict = sorted(max_sent_num_dict.items())
    for key, value in sorted_max_num_dict:
        print(key, value)
    
    print("min_sent_num:")
    sorted_min_num_dict = sorted(min_sent_num_dict.items())
    for key, value in sorted_min_num_dict:
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
