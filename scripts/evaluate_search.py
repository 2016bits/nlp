import json
import argparse
import ast

from utils import log

def get_document(evidence_list):
    documents = []
    for evidence in evidence_list:
        doc_list = []
        for doc, _ in evidence:
            if doc not in doc_list:
                doc_list.append(doc)
        if doc_list not in documents:
            documents.append(doc_list)
    return documents

def equal(gold_list, pred_list):
    matched = True
    for doc in gold_list:
        if doc not in pred_list:
            matched = False
    return matched

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
                total_num += 1

                # 预测的evidence只要匹配一组完整的ground_truth evidence即认为证据预测正确
                for doc_list in gold_doc:
                    if equal(doc_list, pred_doc):
                        acc_num += 1
                        break

                gold_doc_list = [doc for doc_list in gold_doc for doc in doc_list]
                gold_doc_list = list(set(gold_doc_list))
                pred_num += len(pred_doc)
                target_num += len(gold_doc_list)

                for gold in gold_doc_list:
                    for pred in pred_doc:
                        if gold == pred:
                            match_num += 1
                
        except:
            print("id: {}".format(inst['id']))
    return match_num, pred_num, target_num, acc_num, total_num

def evaluate_on_dpr(dataset):
    match_num = 0
    pred_num = 0
    target_num = 0

    acc_num = 0
    total_num = 0

    for inst in dataset:
        try:
            evidence = inst['answers']
            if evidence != "[]" and evidence != "doc":
                total_num += 1

                gold_doc = ast.literal_eval(evidence)
                pred_doc = [item['title'] for item in inst['ctxs']]
            
                # 预测的evidence只要匹配一组完整的ground_truth evidence即认为证据预测正确
                for doc_list in gold_doc:
                    if equal(doc_list, pred_doc):
                        acc_num += 1
                        break

                gold_doc_list = [doc for doc_list in gold_doc for doc in doc_list]
                gold_doc_list = list(set(gold_doc_list))
                pred_num += len(pred_doc)
                target_num += len(gold_doc_list)

                for gold in gold_doc_list:
                    for pred in pred_doc:
                        if gold == pred:
                            match_num += 1
        except:
            print("claim: {}".format(inst['question']))
    return match_num, pred_num, target_num, acc_num, total_num

def main(args):
    log_path = args.log_path + args.dataset + "_evaluate_search_on_" + args.type + ".log"
    logger = log.get_logger(log_path)

    logger.info("loading data......")
    if args.type == 'program':
        data_path = args.program_data_path + args.dataset + args.program_in_path
    else:
        data_path = args.dpr_data_path + args.dpr_top + ".json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    if args.type == 'program':
        match_num, pred_num, target_num, acc_num, total_num = evaluate_on_searched_programs(dataset)
    else:
        match_num, pred_num, target_num, acc_num, total_num = evaluate_on_dpr(dataset)

    precision = float(match_num) / float(pred_num + 1e-6)
    recall = float(match_num) / float(target_num + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    acc = float(acc_num) / float(total_num + 1e-6)
    logger.info("pred_num: {}, target_num: {}, match_num: {}".format(pred_num, target_num, match_num))
    logger.info("precision: {}, recall: {}, f1: {}".format(round(precision, 4), round(recall, 4), round(f1, 4)))
    logger.info("acc_num: {}, total_num: {}, acc: {}".format(acc_num, total_num, round(acc, 4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # search on programs
    parser.add_argument('--type', type=str, default='program', choices=['dpr', 'program'])
    parser.add_argument('--program_data_path', type=str, default='./results/ablation/')
    parser.add_argument('--program_in_path', type=str, default='_test_search_results.json')
    parser.add_argument('--dataset', type=str, default="FEVER")
    parser.add_argument('--log_path', type=str, default='./logs/')

    # search on dpr
    parser.add_argument('--dpr_data_path', type=str, default='./data/Wikipedia/dpr/data/results_')
    parser.add_argument('--dpr_top', type=str, default="20", choices=['10', '20', '50', '100'])

    args = parser.parse_args()
    main(args)
