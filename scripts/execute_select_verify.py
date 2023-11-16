import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

from utils import log
from search_module import Search_wikipage, Search_Wiki

class Program_Execute:
    def __init__(self, args, logger):
        self.model_name = args.model_name
        self.device = args.device

        logger.info(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(args.device)
        # self.model.parallelize()
        logger.info(f"Model {self.model_name} loaded.")

        self.WikiPage = Search_wikipage(args.db_path, args.db_table)
        self.wiki = Search_Wiki(args.db_path, args.db_table, args.num_retrieved)
    
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

    def execute_program(self, dataset, logger):
        # target labels, predicted labels
        target_labels, pred_labels = [], []
        results = []

        for inst in tqdm(dataset):
            target_labels.append(inst['gold_label'])

            # get wikipages according predicted titles
            wiki_pages = self.WikiPage.select_wikipage(inst['parsed_title'])

            # get evidence
            evidence_dict = self.wiki.find_evidences(wiki_pages, inst['parsed_sentence'])

            # verify
            evidence = " ".join(evidence_dict['evidence_texts'])
            input_ids = self.tokenizer(inst["claim"], evidence, return_tensors="pt", truncation=True).to(self.device)
            output = self.model(input_ids['input_ids'])
            prediction = torch.softmax(output["logits"][0], -1)
            prob, pred = torch.max(prediction, dim=-1)
            prob = prob.item()
            
            evidences = evidence_dict['evidence_ids']

            # get results
            if pred == 0:
                final_label = 'REFUTES'
            elif pred == 1:
                final_label = 'SUPPORTS'
            elif pred == 2:
                final_label = 'NOT ENOUGH INFO'

            pred_labels.append(final_label)
            results.append({
                'id': inst['id'],
                'claim': inst['claim'],
                'gold_label': inst['gold_label'],
                'gold_evidence': inst['gold_evidence'],
                'pred_label': final_label,
                'pred_probability': prob,
                'pred_evidence': evidences
            })
        
        # evaluate
        self.evaluate(pred_labels, target_labels, logger, 3)

        return results

def main(args):
    # data path
    # in_path = args.in_path + args.dataset + "_" + args.mode + "_merge_parse_search_claim_results.json"
    in_path = args.in_path + args.dataset + "_" + args.mode + "_merge_parse_search_keyword_results.json"
    out_path = args.out_path + args.dataset + "_" + args.mode + "_results.json"
    log_path = args.log_path + args.dataset + "_" + args.mode + "_execute_select_verify.log"

    # init log
    logger = log.get_logger(log_path)
    logger.info(args)

    # load data
    logger.info('load data......')
    with open(in_path, 'r') as f1:
        dataset = json.load(f1)
    
    # load model
    logger.info("loading model......")
    program_execute = Program_Execute(args, logger)
    results = program_execute.execute_program(dataset, logger)

    # finish prediction
    logger.info("prediction finished...")
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--in_path', type=str, default='./results/parse/')
    parser.add_argument('--dataset', type=str, default='FEVER')
    parser.add_argument('--search_method', type=str, default='mediawiki')
    parser.add_argument('--top_doc', type=int, default=3)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev'])
    parser.add_argument('--out_path', type=str, default='./results/fact_checking/')
    parser.add_argument('--log_path', type=str, default='./logs/')
    
    # wikipedia arguments
    parser.add_argument('--db_path', type=str, default='./data/Wikipedia/data/wikipedia.db')
    parser.add_argument('--db_table', type=str, default='documents')

    # model arguments
    # parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    # parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
    parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    # training arguments
    parser.add_argument('--num_retrieved', type=int, default=5)

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda:1")

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    main(args)
