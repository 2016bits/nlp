import json
import argparse
import torch
import sys

from fever.scorer import fever_score
from prettytable import PrettyTable

def evaluate(predictions, gold_data_path):
    predicted_labels =[]
    predicted_evidence = []
    actual = []
    ids = dict()
    
    for pred in predictions:
        ids[pred["id"]] = len(predicted_labels)
        predicted_labels.append(pred["pred_label"])
        predicted_evidence.append(pred["pred_evidence"])
        actual.append(0)

    with open(gold_data_path, "r") as actual_file:
        for line in actual_file:
            id = json.loads(line)["id"]
            if id in ids:
                actual[ids[id]] = json.loads(line)

    predictions = []
    for ev,label in zip(predicted_evidence, predicted_labels):
        predictions.append({"predicted_evidence":ev,"predicted_label":label})

    score,acc,precision,recall,f1 = fever_score(predictions,actual)

    tab = PrettyTable()
    tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
    tab.add_row((round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

    print(tab)

def main(args):
    log_path = args.log_path + args.dataset_name + '_evaluate_with_chatgpt.log'
    sys.stdout = open(log_path, mode='w')
    
    # load data
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
            
    # convert predictions and gold for calculating fever score
    evaluate(dataset, args.gold_data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--log_path', type=str, default='./logs/')
    # parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--data_path', type=str, default='./results/ablation/FEVER_test_verify_with_chatgpt.json')
    parser.add_argument('--gold_data_path', type=str, default='./data/FEVER/SelectData/dev_eval.json')
    parser.add_argument('--dataset_name', type=str, default='FEVER', choices=['SCIFACT'])
    parser.add_argument('--save_path', type=str, default='./results/ablation/')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev', 'test'])

    args = parser.parse_args()

    main(args)
