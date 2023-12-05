import json
import argparse
from tqdm import tqdm

def equal(gold_list, pred_list):
    matched = True
    for doc in gold_list:
        if doc not in pred_list:
            matched = False
    return matched

def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["gold_label"].upper() != "NOT ENOUGH INFO":
        all_evi = [e for eg in instance["gold_evidence"] for e in eg if e is not None]
        predicted_evidence = instance["pred_evidence"] if max_evidence is None else \
            instance["pred_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            pred = [prediction[0], prediction[1]]
            if pred in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["gold_label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["gold_evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        predicted_evidence = [[e[0], e[1]] for e in instance['pred_evidence']]

        for evidence in instance["gold_evidence"]:
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0

def main(args):
    with open(args.result_path, 'r') as f:
        results = json.load(f)
    
    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    for instance in tqdm(results):
        macro_prec = evidence_macro_precision(instance)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0
    f1 = 2.0 * pr * rec / (pr + rec)

    print("macro_precision: {}, macro_precision_hits: {}, pr: {}".format(macro_precision, macro_precision_hits, pr))
    print("macro_recall: {}, macro_recall_hits: {}, rec: {}".format(macro_recall, macro_recall_hits, round(rec, 4)))
    print("f1: {}".format(round(f1, 4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='./results/select/select_bert_contrastive.json')

    args = parser.parse_args()
    main(args)
