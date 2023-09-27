import argparse
import torch
import json
import sqlite3
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import log, flagai_model

label_map = {
    'no': 0,
    'false': 0,
    'yes': 1,
    'true': 1,
    'unknown': 2,
    'un': 2
}

class Verify:
    def __init__(self, args, logger):
        self.dataset_name = args.dataset_name
        self.db_table = args.db_table
        self.max_t5_tokens = args.max_t5_tokens
        self.device = args.device

        conn = sqlite3.connect(args.db_path)
        self.c = conn.cursor()
        logger.info("connect to database successfully...")

        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        logger.info("load tokenizer successfully...")

        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.model_path).to(args.device)
        logger.info("load model successfully...")

    def convert_evidence(self, evidence_list):
        evidence_text = ""
        for title, line_id in evidence_list:
            sql = """select * from {} where id = "{}";""".format(self.db_table, title)
            cursor = self.c.execute(sql)
            for row in cursor:
                lines = row[2].split('\n')
                for line in lines:
                    sent_id = eval(line.split('\t')[0])
                    if sent_id == line_id:
                        sent_text = line.replace('{}\t'.format(sent_id), '')
                        evidence_text += sent_text
        return evidence_text
    
    def evaluate(self, predictions, ground_truth, logger, num_of_classes=2):
        if num_of_classes == 2:
            target_names = ['REFUTES', 'SUPPORTS']
            label_map = {'REFUTES': 0, 'SUPPORTS': 1}
            gold = [label_map[e] for e in ground_truth]
            pred = [label_map[e] for e in predictions]
            logger.info(classification_report(gold, pred, labels=range(2), target_names=target_names, digits=4))
            logger.info(confusion_matrix(gold, pred))
        elif num_of_classes == 3:
            target_names = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']
            label_map = {'REFUTES': 0, 'SUPPORTS': 1, 'NOT ENOUGH INFO': 2}
            gold = [label_map[e] for e in ground_truth]
            pred = [label_map[e] for e in predictions]
            logger.info(classification_report(gold, pred, labels=range(3), target_names=target_names, digits=4))
            logger.info(confusion_matrix(gold, pred))
    
    def verify(self, logger, datasets):
        results = []

        predictions = []
        targets = []
        logger.info("Start predicting...")
        
        for inst in tqdm(datasets):
            claim = inst['claim']
            evidence = self.convert_evidence(inst['evidence'])
            target = inst['label']
            targets.append(target)

            # create prompt
            input_text = f"{evidence}\nBased on the above information, is it true that {claim}? true, false or unknown? The answer is: "
            if len(input_text) > self.max_t5_tokens:
                evidence_len = self.max_t5_tokens - len(claim) - 25
                input_text = f"{evidence[:evidence_len]}\nBased on the above information, is it true that {claim}? true, false or unknown? The answer is: "

            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model.generate(input_ids, max_length=None, max_new_tokens=8)
            pred = self.tokenizer.batch_decode(res, skip_special_tokens=True)[0].strip()

            # map output to label_map
            pred = pred.lower().strip()
            label_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'unknown': 2, 'un': 2, 
                        "it's impossible to say": 2, 'not enough info': 2, 'not enough information': 2}
            if pred in label_map:
                pred = label_map[pred]
            else:
                logger.info("Alert! prediction error id:{}, prediction: {}".format(inst['id'], pred))
                pred = random.sample([0, 1, 2], 1)[0]
            
            if pred == 0:
                pred_label = 'REFUTES'
            elif pred == 1:
                pred_label = 'SUPPORTS'
            elif pred == 2:
                pred_label = 'NOT ENOUGH INFO'
            predictions.append(pred_label)
            
            results.append({
                'id': inst['id'],
                'claim': inst['claim'],
                'label': inst['label'],
                'prediction': pred_label
            })

        # evaluate
        self.evaluate(predictions, targets, logger, 3)

        return results

def main(args):
    log_path = args.log_path + args.dataset_name + '_verify_with_finetuned_t5.log'

    # init logger
    logger = log.get_logger(log_path)
    logger.info(args)

    # load data
    logger.info("loading data......")
    data_path = args.data_path + args.dataset_name + "/processed/" + args.mode + ".json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    # load model
    logger.info("loading model......")
    model = Verify(args, logger)
    outputs = model.verify(logger, dataset)

    # save outputs
    save_path = args.save_path + args.dataset_name + "_test_verify_with_finetuned_t5_results.json"
    with open(save_path, 'w') as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    logger.info("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='FEVER', choices=['SCIFACT'])
    parser.add_argument('--save_path', type=str, default='./results/ablation/')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev', 'test'])

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda:1")

    # wikipedia arguments
    parser.add_argument('--db_path', type=str, default='./data/Wikipedia/data/wikipedia.db')
    parser.add_argument('--db_table', type=str, default='documents')

    # model
    parser.add_argument('--max_t5_tokens', type=int, default=1024)
    parser.add_argument('--model_name', type=str, default='google/flan-t5-xl')
    parser.add_argument('--cache_dir', type=str, default='./google/flan-t5-xl')
    parser.add_argument('--model_path', type=str, default='./model/finetuned_t5_FEVER_100shot_train_data.pth')
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    main(args)
