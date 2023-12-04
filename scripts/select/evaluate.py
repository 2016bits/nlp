import json
import argparse

def equal(gold_list, pred_list):
    matched = True
    for doc in gold_list:
        if doc not in pred_list:
            matched = False
    return matched

def evaluate_select(dataset):
    match_num = 0
    pred_num = 0
    target_num = 0

    acc_num = 0
    total_num = 0

    for inst in dataset:
        try:
            gold_evidence = inst['gold_evidence']
            pred_evidence = inst['pred_evidence']
            
            if inst['gold_label'] != 'NOT ENOUGH INFO':
                total_num += 1
                matched = False
                max_pred_num = len(pred_evidence)
                max_target_num = 0
                max_match_num = 0
                max_f1 = 0

                # 预测的evidence只要匹配一组完整的ground_truth evidence即认为证据预测正确
                for evidence_list in gold_evidence:
                    if not matched and equal(evidence_list, pred_evidence):
                        acc_num += 1
                        matched = True
                    intersect_num = 0
                    for pred in pred_evidence:
                        for gold in evidence_list:
                            if pred == gold:
                                intersect_num += 1
                    p = float(intersect_num) / float(max_pred_num + 1e-6)
                    r = float(intersect_num) / float(len(evidence_list) + 1e-6)
                    f1 = 2 * p * r / (p + r + 1e-6)
                    if f1 > max_f1:
                        max_f1 = f1
                        max_target_num = len(evidence_list)
                        max_match_num = intersect_num

                if max_match_num == 0:
                    max_target_num = min(len(evidence) for evidence in gold_evidence)
                match_num += max_match_num
                pred_num += max_pred_num
                target_num += max_target_num
                
        except:
            print("id: {}".format(inst['id']))
    return match_num, pred_num, target_num, acc_num, total_num


def main(args):
    with open(args.result_path, 'r') as f:
        results = json.load(f)
    
    match_num, pred_num, target_num, acc_num, total_num = evaluate_select(results)
    precision = float(match_num) / float(pred_num + 1e-6)
    recall = float(match_num) / float(target_num + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    acc = float(acc_num) / float(total_num + 1e-6)
    print("pred_num: {}, target_num: {}, match_num: {}".format(pred_num, target_num, match_num))
    print("precision: {}, recall: {}, f1: {}".format(round(precision, 4), round(recall, 4), round(f1, 4)))
    print("acc_num: {}, total_num: {}, acc: {}".format(acc_num, total_num, round(acc, 4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='./results/select/select_bert_contrastive.json')

    args = parser.parse_args()
    main(args)
