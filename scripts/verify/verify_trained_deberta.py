import argparse
import torch
import json
import sqlite3
import re
from tqdm import tqdm
from fever.scorer import fever_score
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import log

label_dict = {
    'SUPPORTS': 0,
    'NOT ENOUGH INFO': 1,
    'REFUTES': 2
}

class Verify:
    def __init__(self, args, logger):
        self.dataset_name = args.dataset_name
        self.db_table = args.db_table
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.device = args.device

        conn = sqlite3.connect(args.db_path)
        self.c = conn.cursor()
        logger.info("connect to database successfully...")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        logger.info("load tokenizer successfully...")

        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name, cache_dir=args.checkpoint).to(args.device)
        logger.info("load model successfully...")
    
    def process_sent(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title

    def convert_evidence(self, evidence_list):
        text = ""
        pred_list = []
        for evidence in  evidence_list:
            pred_list.append([evidence[0], evidence[1]])
            if evidence[3] > 0:
                text += "<title> {} <sentence> {} ".format(self.process_wiki_title(evidence[0]), self.process_sent(evidence[2]))
        return text, pred_list
    
    def batch_data(self, examples):
        indexs = []
        labels = []
        ids = []
        msks = []
        evidences = []
        for inst in examples:
            claim = inst['claim']
            evidence, pred_evidence = self.convert_evidence(inst['evidence'])
            text = "<claim> {} <evidence> {}".format(claim, evidence)
            label = label_dict[inst['label']]

            emebddings = self.tokenizer.encode_plus(text, truncation=True, padding='max_length', max_length=self.max_len)

            indexs.append(inst['id'])
            labels.append(label)
            ids.append(emebddings['input_ids'])
            msks.append(emebddings['attention_mask'])
            evidences.append(pred_evidence)
        
        id_tensors = torch.LongTensor(ids).to(self.device)
        msk_tensors = torch.LongTensor(msks).to(self.device)
        label_ids = torch.LongTensor(labels).to(self.device)
        return id_tensors, msk_tensors, label_ids, indexs, evidences
    
    def batch_list(self, dataset):
        data_num = len(dataset)
        total_batch_num = data_num // self.batch_size + 1 if data_num % self.batch_size != 0 else data_num // self.batch_size
        batched_data = []
        for index in range(total_batch_num):
            examples = dataset[index*self.batch_size: (index+1)*self.batch_size]
            batched_data.append(self.batch_data(examples))
        return batched_data
        
    def verify(self, logger, batched_data):
        pred_results = []
        save_results = {}

        logger.info("Start predicting...")
        
        for inst in tqdm(batched_data):
            ids, msks, labels, indexs, pred_evidences = inst
            outputs = self.model(input_ids=ids, attention_mask=msks, labels=labels)
            _, preds = torch.max(outputs.logits, 1)

            for index in range(len(preds)):
                pred = preds[index]
                if pred == 2:
                    pred_label = 'REFUTES'
                elif pred == 0:
                    pred_label = 'SUPPORTS'
                elif pred == 1:
                    pred_label = 'NOT ENOUGH INFO'
                
                pred_results.append({
                    'id': indexs[index],
                    'pred_label': pred_label,
                    'pred_evidence': pred_evidences[index],
                })
                save_results[indexs[index]] = {
                    'id': indexs[index],
                    'pred_label': pred_label,
                    'pred_evidence': pred_evidences[index],
                }
                
        return pred_results, save_results

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
            actual[ids[json.loads(line)["id"]]] = json.loads(line)

    predictions = []
    for ev,label in zip(predicted_evidence, predicted_labels):
        predictions.append({"predicted_evidence":ev,"predicted_label":label})

    score,acc,precision,recall,f1 = fever_score(predictions,actual)

    tab = PrettyTable()
    tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
    tab.add_row((round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

    print(tab)

def main(args):
    log_path = args.log_path + args.dataset_name + '_verify_with_finetuned_DeBERTa_large.log'

    # init logger
    logger = log.get_logger(log_path)
    logger.info(args)

    # load data
    logger.info("loading data......")
    # data_path = args.data_path + args.dataset_name + "/processed/" + args.mode + ".json"
    dataset = []
    with open(args.data_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    
    # load model
    logger.info("loading model......")
    model = Verify(args, logger)
    batched_data = model.batch_list(dataset)
    pred_results, save_results = model.verify(logger, batched_data)

    # save outputs
    outputs = []
    with open(args.gold_data_path, 'r') as f:
        for line in f:
            inst = json.loads(line)
            id = inst['id']
            data = save_results[id]
            outputs.append({
                'id': id,
                'claim': inst['claim'],
                'gold_evidence': inst['evidence'],
                'gold_label': inst['label'],
                'pred_evidence': data['pred_evidence'],
                'pred_label': data['pred_label']
            })
            
    save_path = args.save_path + args.dataset_name + "_test_verify_bever_with_finetuned_DeBERTa_large_results.json"
    with open(save_path, 'w') as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    logger.info("Finished!")

    # convert predictions and gold for calculating fever score
    evaluate(pred_results, args.gold_data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--log_path', type=str, default='./logs/')
    # parser.add_argument('--data_path', type=str, default='./data/')

    # KenerGAT得到的selected_evidence数据
    # parser.add_argument('--data_path', type=str, default='./data/FEVER/SelectData/bert_eval.json')

    # BEVER得到的selected_evidence数据
    parser.add_argument('--data_path', type=str, default='data/FEVER/BeverData/dev_selected_evidence.jsonl')

    parser.add_argument('--gold_data_path', type=str, default='./data/FEVER/SelectData/dev_eval.jsonl')
    parser.add_argument('--dataset_name', type=str, default='FEVER', choices=['SCIFACT'])
    parser.add_argument('--save_path', type=str, default='./results/ablation/')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev', 'test'])

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")

    # wikipedia arguments
    parser.add_argument('--db_path', type=str, default='./data/Wikipedia/data/wikipedia.db')
    parser.add_argument('--db_table', type=str, default='documents')

    # model
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=512)
    # parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    # parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
    parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    parser.add_argument('--checkpoint', type=str, default='./models/deberta_verify_bever.pth')
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    main(args)
