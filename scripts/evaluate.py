import json
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utils import log

def is_strictly_correct(inst, max_evidence=None):
    if inst['pred_label'].upper() != inst['gold_label'].upper():
        return False
    
    elif inst['pred_label'].upper == "NOT ENOUGH INFO":
        return True
    
    else:
        if max_evidence == None:
            max_evidence = len(inst['pred_evidence'])
        
        for evidence in inst['gold_evidence']:
            if evidence not in inst['pred_evidence']:
                return False
        return True

def evaluate_with_evidence(results, logger, max_evidence=5):
    correct = 0
    strict = 0
    total = 0
    for inst in results:
        total += 1
        
        if inst['pred_label'] == inst['gold_label']:
            correct += 1
        if is_strictly_correct(inst, max_evidence):
            strict += 1
    
    fever_score = strict / total
    acc = correct / total

    logger.info("FEVER score: {}, Accuracy: {}".format(round(fever_score, 4), round(acc, 4)))


def evaluate(predictions, ground_truth, logger, num_of_classes=2):
    if num_of_classes == 2:
        target_names = ['REFUTES', 'SUPPORTS']
        label_map = {'REFUTES': 0, 'SUPPORTS': 1}
        gold = [label_map[e] for e in ground_truth]
        pred = [label_map[e] for e in predictions]
        logger.info(classification_report(gold, pred, labels=range(2), target_names=target_names, digits=4, output_dict=True))
        logger.info(confusion_matrix(gold, pred))
    elif num_of_classes == 3:
        target_names = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']
        label_map = {'REFUTES': 0, 'SUPPORTS': 1, 'NOT ENOUGH INFO': 2}
        gold = [label_map[e] for e in ground_truth]
        pred = [label_map[e] for e in predictions]
        logger.info(classification_report(gold, pred, labels=range(3), target_names=target_names, digits=4))
        logger.info(confusion_matrix(gold, pred))

def main(args):
    logger = log.get_logger(args.log_path)
    with open(args.result_path, 'r') as f:
        results = json.load(f)
    evaluate_with_evidence(results, logger, args.max_evidence)

    # target_labels = []
    # prediction_labels = []
    # count = 0
    # acc_num = 0
    # for result in results:
    #     target = result["gold_label"]
    #     pred = result["pred_label"]
    #     target_labels.append(target)
    #     prediction_labels.append(pred)

    #     count += 1
    #     if target == pred:
    #         acc_num += 1
    
    # evaluate(prediction_labels, target_labels, logger, num_of_classes=3)
    # acc = acc_num / count
    # logger.info("sum: {}, acc_num: {}, acc: {}".format(count, acc_num, acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_evidence', type=int, default=5)
    parser.add_argument('--result_path', type=str, default='./results/fact_checking/FEVER_test_results.json')
    parser.add_argument('--log_path', type=str, default='./logs/FEVER_test_result_with_evidence.log')
    # parser.add_argument('--result_path', type=str, default='./outputs/execute/2023-07-31_20:43:38/results/fact_checking/FEVER_test_results.json')
    # parser.add_argument('--log_path', type=str, default='./logs/FEVER_test_result_with_evidence2.log')
    # parser.add_argument('--result_path', type=str, default='./results/ablation/FEVER_test_only_t5_results.json')
    # parser.add_argument('--log_path', type=str, default='./logs/FEVER_test_only_t5_result.log')

    args = parser.parse_args()
    main(args)
