import argparse
import json
from tqdm import tqdm
from search_module import Search_Tfidf

def main(args):
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    tfidf = Search_Tfidf(args.db_path, args.max_page, args.max_sent, args.tfidf_model)
    results = []
    for data in tqdm(dataset):
        if data['label'] == 'NOT ENOUGH INFO':
            evidence_dict = tfidf.search_sents(data['claim'])
            results.append({
                'id': data['id'],
                'claim': data['claim'],
                'evidence': evidence_dict
            })
    with open(args.save_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_page', type=int, default=3)
    parser.add_argument('--max_sent', type=int, default=5)
    parser.add_argument('--db_path', type=str, default='./data/Wikipedia/data/wikipedia.db')
    parser.add_argument('--tfidf_model', type=str, default='./data/Wikipedia/data/tfidf.npz')

    parser.add_argument('--data_path', type=str, default='./data/FEVER/test_data.json')
    parser.add_argument('--save_path', type=str, default='./scripts/test/nei.json')

    args = parser.parse_args()
    main(args)
