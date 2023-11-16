import argparse
import json

def main(args):
    # data path
    # parsed_result_path = args.parsed_result_path + args.dataset + "_" + args.mode + "_parse_results.json"
    # searched_result_path = args.searched_result_path + args.dataset + "_" + args.mode + "_" + args.search_method + "_search_claim_" + str(args.top_doc) + ".jsonl"
    # out_path = args.output_path + args.dataset + "_" + args.mode + "_merge_parse_search_claim_results.json"
    parsed_result_path = args.parsed_result_path + args.dataset + "_" + args.mode + "_parse_results.json"
    searched_result_path = args.searched_result_path + args.dataset + "_" + args.mode + "_" + args.search_method + "_search_keyword_" + str(args.top_doc) + ".jsonl"
    out_path = args.output_path + args.dataset + "_" + args.mode + "_merge_parse_search_keyword_results.json"
    
    # load data
    print("load data......")
    with open(parsed_result_path, 'r') as f1:
        parsed_results = json.load(f1)
    searched_results = []
    with open(searched_result_path, 'r') as f2:
        for line in f2.readlines():
            searched_results.append(json.loads(line.strip()))
    
    print("finish loading data and start reconstructing searched_results to dict......")
    # reconstruct searched_results to dict(key: id, value: predicted_pages)
    search_dict = {}
    for data in searched_results:
        search_dict[data['id']] = data['predicted_pages']
    
    print("merge data......")
    merge_data = []
    for data in parsed_results:
        predicted_pages = search_dict[data['id']]
        merge_data.append({
            'id': data['id'],
            'claim': data['claim'],
            'gold_label': data['gold_label'],
            'gold_evidence': data['gold_evidence'],
            'parsed_title': predicted_pages,
            'parsed_sentence': data['parsed_sentence']
        })
    
    print(len(merge_data))
    print("finish merging and start writing......")
    with open(out_path, 'w') as f3:
        f3.write(json.dumps(merge_data, indent=2))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--parsed_result_path', type=str, default='./results/parse/')
    parser.add_argument('--searched_result_path', type=str, default='./results/search/')
    parser.add_argument('--dataset', type=str, default='FEVER')
    parser.add_argument('--search_method', type=str, default='mediawiki')
    parser.add_argument('--top_doc', type=int, default=5)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev'])
    parser.add_argument('--output_path', type=str, default='./results/parse/')
    
    args = parser.parse_args()
    main(args)
