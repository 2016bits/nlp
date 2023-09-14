import argparse
import torch
import json
import random
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, confusion_matrix

from utils.data import RawData
from utils import log


class Verify_Only_T5:
    def __init__(self, args, logger):
        self.batch_size = args.batch_size
        self.device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu >=0 else "cpu"
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name

        logger.info(f"Loading model {self.model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(self.device)

        logger.info(f"Model {self.model_name} loaded.")
    
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
    
    def verify(self, dataset, logger):
        logger.info("start verifying......")
        results = []
        predictions = []
        targets = []
        # batched_data = [dataset[i: i+self.batch_size] for i in range(0, len(dataset), self.batch_size)]
        for inst in tqdm(dataset):
            claim = inst['claim']
            target = inst['label']
            targets.append(target)

            # generate with t5
            input_text = f"Is it true that {claim}? True, false or not enough information? The answer is: "

            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model.generate(input_ids, max_length=None, max_new_tokens=8)
            pred = self.tokenizer.batch_decode(res, skip_special_tokens=True)[0].strip()

            # map output to label_map
            pred = pred.lower().strip()
            label_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, "it's impossible to say": 2, 'not enough info': 2, 'not enough information': 2}
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
    log_path = args.log_path + args.dataset_name + '_test_only_t5_result.log'

    # init logger
    logger = log.get_logger(log_path)
    logger.info(args)

    # load data
    logger.info("loading data......")
    test_data_path = args.data_path + args.dataset_name + "/test_data.json"
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    test_data = test_data if args.num_eval_samples < 0 else test_data[:args.num_eval_samples]

    # load model
    logger.info("loading model......")
    model = Verify_Only_T5(args, logger)
    results = model.verify(test_data, logger)

    # finish prediction
    logger.info("prediction finished...")
    out_path = args.save_path + args.dataset_name + "_test_only_t5_results.json"
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='FEVER', choices=['SCIFACT'])
    parser.add_argument('--save_path', type=str, default='./results/ablation/')
    parser.add_argument('--model_name', type=str, default='google/flan-t5-xl')
    parser.add_argument('--cache_dir', type=str, default="./google/flan-t5-xl")
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--num_eval_samples', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()

    if torch.cuda.is_available and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))

    main(args)
