import json
import argparse

from utils import log

def get_document(evidence_list):
    documents = []
    for evidence in evidence_list:
        for doc, _ in evidence:
            if doc not in documents:
                documents.append(doc)
    return documents

def evaluate_on_searched_programs(dataset):
    match_num = 0
    pred_num = 0
    target_num = 0

    acc_num = 0
    total_num = 0

    for inst in dataset:
        try:
            evidence = inst['gold_evidence']
            gold_doc = get_document(evidence)
            pred_doc = inst['searched_document']
            
            if inst['gold_label'] != 'NOT ENOUGH INFO':
                pred_num += len(pred_doc)
                target_num += len(gold_doc)

                for gold in gold_doc:
                    for pred in pred_doc:
                        if gold == pred:
                            match_num += 1

                total_num += 1
                if pred_doc == gold_doc:
                    acc_num += 1
        except:
            print("id: {}".format(inst['id']))
    return match_num, pred_num, target_num, acc_num, total_num

def main(args):
    log_path = args.log_path + args.dataset + "_evaluate_search.log"
    logger = log.get_logger(log_path)

    logger.info("loading data......")
    data_path = args.data_path + args.dataset + args.in_path
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    match_num, pred_num, target_num, acc_num, total_num = evaluate_on_searched_programs(dataset)

    precision = float(match_num) / float(pred_num + 1e-6)
    recall = float(match_num) / float(target_num + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    acc = float(acc_num) / float(total_num + 1e-6)
    logger.info("pred_num: {}, target_num: {}, match_num: {}".format(pred_num, target_num, match_num))
    logger.info("precision: {}, recall: {}, f1: {}".format(round(precision, 4), round(recall, 4), round(f1, 4)))
    logger.info("acc_num: {}, total_num: {}, acc: {}".format(acc_num, total_num, round(acc, 4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./results/ablation/')
    parser.add_argument('--in_path', type=str, default='_test_search_results.json')
    parser.add_argument('--dataset', type=str, default="FEVER")
    parser.add_argument('--log_path', type=str, default='./logs/')

    args = parser.parse_args()
    main(args)
