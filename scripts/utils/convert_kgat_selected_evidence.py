import argparse
import json

def main(args):
    # 读取BEVER生成的selected_evidence数据
    top5_sentences = []
    with open(args.top5_sentence_path, 'r') as f:
        for line in f:
            top5_sentences.append(json.loads(line.strip()))

    # 读取gold数据
    gold_dataset = []
    with open(args.gold_path, 'r') as f:
        for line in f:
            gold_dataset.append(json.loads(line.strip()))

    # 将selected_evidence数据和gold数据合并
    gold_data_dict = {}
    for data in gold_dataset:
        id = data['id']
        gold_data_dict[id] = data

    results = []    
    for top5 in top5_sentences:
        id = top5['id']
        evidence = gold_data_dict[id]
        results.append({
            "id": id,
            "claim": top5['claim'],
            "gold_label": top5['label'],
            "gold_evidence": evidence,
            "pred_evidence": top5['evidence']
        })

    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top5_sentence_path', type=str, default="data/FEVER/BeverData/train_selected_evidence.jsonl")
    parser.add_argument('--gold_path', type=str, default="./data/FEVER/SelectData/all_train.json")
    parser.add_argument('--save_path', type=str, default="./data/FEVER/BeverData/deberta_train.json")
    args = parser.parse_args()
    main(args)
