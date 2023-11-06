import torch
import argparse
import re
import json
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

from utils import log, data
from search_module import Search_Wiki, Search_Tfidf


class Program_Execute:
    def __init__(self, args, logger):
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.max_deberta_tokens = args.max_deberta_tokens
        self.device = args.device

        logger.info(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(args.device)
        # self.model.parallelize()
        logger.info(f"Model {self.model_name} loaded.")

        self.tfidf = Search_Tfidf(args.db_path, args.max_page, args.max_sent, args.tfidf_model)
        self.wiki = Search_Wiki(args.db_path, args.db_table, args.num_retrieved)
    
    # parse command to get command type
    def get_command_type(self, command):
        if command.find("claim =") >= 0:
            return "DEFINITE"
        elif command.find("= Search_pages") >= 0:
            return "SEARCH"
        elif command.find("= Find_evidences") >= 0:
            return "FIND"
        elif command.find("= Verify") >= 0:
            return "VERIFY"
        else:
            return "UNKNOWN"
        
    # parse definite command to get claim
    def parse_definite_command(self, command):
        var_name, var_value = command.split('=')
        var_name = var_name.strip()
        var_value = var_value.strip().strip('"')
        return var_name, var_value
    
    # parse search command to get variable and parameters
    def parse_search_command(self, command, variable_map):
        # after executing search command, get variable
        return_var, tmp = command.split('= Search_pages')
        return_var = return_var.strip()

        # get parameter of search function
        pattern = re.compile(f'Search_pages\(\"(.*)\"\)', re.S)
        matching = re.findall(pattern, command)
        texts = matching if len(matching) > 0 else tmp

        # replace variable
        for var_name, var_value in variable_map.items():
            for text in texts:
                if text == var_name:
                    text = var_value

        return return_var, texts

    # parse find command to get variable and parameters
    def parse_find_command(self, command, variable_map):
        return_var, tmp = command.split('= Find_evidences')
        return_var = return_var.strip()

        pattern = r'Find_evidences\((.+?),(.+)\)'
        matching = re.search(pattern, command)
        param1 = matching.group(1)
        param2 = matching.group(2)
        params = re.findall(r'"(.+?)"', param2)
        key_words = params if len(params) > 0 else tmp

        # replace variable
        for var_name, var_value in variable_map.items():
            if param1 == var_name:
                param1 = var_value
        
        return return_var, param1, key_words

    # parse verify command to get variable and parameters
    def parse_verify_command(self, command, variable_map, logger):
        return_var, _ = command.split('= Verify')
        return_var = return_var.strip()

        pattern = re.compile(f'Verify\((.*)\)', re.S)
        command_args = re.findall(pattern, command)
        verify_subs = command_args[0].split(",")
        claim, evidences = [param.strip() for param in verify_subs]

        if claim in variable_map:
            claim = variable_map[claim]
        else:
            logger.info("Alert!!! wrong parameter: {}".format(claim))
            return return_var, claim, evidences, False
        if evidences in variable_map:
            evidences = variable_map[evidences]
        else:
            logger.info("Alert!!! wrong parameter: {}".format(evidences))
            return return_var, claim, evidences, False
        
        return return_var, claim, evidences, True
    
    # parse program to execute
    def parse_program(self, id, program, logger):
        # map variable to value
        variable_map = {}
        
        # parse command line by line
        for command in program:
            cmd_type = self.get_command_type(command)

            # definite variable
            if cmd_type == "DEFINITE":
                var_name, var_value = self.parse_definite_command(command)
                variable_map[var_name] = var_value

            # search relevant wikipedia pages
            elif cmd_type == "SEARCH":
                # complete ) and "
                quote_num = len(re.findall(r'"', command))
                if quote_num % 2:
                    command += '"'
                if command[-1] != ')':
                    command += ')'

                return_var, text = self.parse_search_command(command, variable_map)
                wikipages = self.wiki.search_wikipages(text)
                
                variable_map[return_var] = wikipages

            # find relevant evidences
            elif cmd_type == "FIND":
                # complete ) and "
                quote_num = len(re.findall(r'"', command))
                if quote_num % 2:
                    command += '"'
                if command[-1] != ')':
                    command += ')'

                return_var, param1, key_words = self.parse_find_command(command, variable_map)
                # evidence dict: {'evidence_ids': _, 'evidence_texts': _}
                evidence_dict = self.wiki.find_evidences(param1, key_words)
                
                variable_map[return_var] = evidence_dict

            # verify the claim
            elif cmd_type == "VERIFY":
                try:
                    return_var, claim, evidence_dict, flag = self.parse_verify_command(command, variable_map, logger)
                    evidence = " ".join(evidence_dict['evidence_texts'])
                    if flag:
                        if len(evidence.split()) + len(claim.split()) > self.max_deberta_tokens:
                            evidence_len = self.max_deberta_tokens - len(claim.split())
                            evidence = " ".join(evidence.split()[:evidence_len])
                        
                        input_ids = self.tokenizer(claim, evidence, return_tensors="pt").to(self.device)
                        output = self.model(input_ids['input_ids'])
                        prediction = torch.softmax(output["logits"][0], -1)
                        prob, pred = torch.max(prediction, dim=-1)
                        prob = prob.item()
                        
                        evidences = evidence_dict['evidence_ids']
                    else:
                        pred = random.sample([0, 1, 2], 1)[0]
                        prob = 0.3333
                        evidences = []
                except:
                    logger.info(f"Alert!!! parsing error: {id}")
                    pred = random.sample([0, 1, 2], 1)[0]
                    prob = 0.3333
                    evidences = []
        if variable_map['claim'] and evidence_dict:
            try:
                evidence = " ".join(evidence_dict['evidence_texts'])
                
                if len(evidence.split()) + len(claim.split()) > self.max_deberta_tokens:
                    evidence_len = self.max_deberta_tokens - len(claim.split())
                    evidence = " ".join(evidence.split()[:evidence_len])
                
                input_ids = self.tokenizer(claim, evidence, return_tensors="pt").to(self.device)
                output = self.model(input_ids['input_ids'])
                prediction = torch.softmax(output["logits"][0], -1)
                prob, pred = torch.max(prediction, dim=-1)
                prob = prob.item()
                
                evidences = evidence_dict['evidence_ids']
                
            except:
                logger.info(f"Alert!!! parsing error: {id}")
                pred = random.sample([0, 1, 2], 1)[0]
                prob = 0.3333
                evidences = []

        return prob, pred, evidences
    
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

    def execute_program(self, logger, dataset):
        # execute predicted programs to get predicted answers
        logger.info("start executing programs......")

        # target labels, predicted labels
        target_labels, pred_labels = [], []
        results = []

        # for each claim
        # id_list = [305, 637, 910]
        for inst in tqdm(dataset):
            # if inst['id'] not in id_list:
                # continue
            programs = inst['predicted_program']
            target_labels.append(inst['label'])

            predicted_inst_labels = []
            # for each predict_program, select the most
            for program in programs:
                try:
                    prob, pred, evidences = self.parse_program(inst['id'], program, logger)
                except:
                    logger.info("Alert! execution error {}".format(inst['id']))
                    # randomly select True or False
                    pred = random.sample([0, 1, 2], 1)[0]
                    prob = 0.3333
                    evidences = []
                predicted_inst_labels.append(pred)
            
            true_count = len([pred for pred in predicted_inst_labels if pred == 0])
            false_count = len([pred for pred in predicted_inst_labels if pred == 2])
            nei_count = len([pred for pred in predicted_inst_labels if pred == 1])
            counts = [false_count, true_count, nei_count]
            max_count = max(counts)
            max_index = counts.index(max_count)
            if max_index == 0:
                final_label = 'REFUTES'
            elif max_index == 1:
                final_label = 'SUPPORTS'
            elif max_index == 2:
                final_label = 'NOT ENOUGH INFO'

            pred_labels.append(final_label)
            results.append({
                'id': inst['id'],
                'claim': inst['claim'],
                'gold_label': inst['label'],
                'gold_evidence': inst['evidence'],
                'pred_label': final_label,
                'pred_probability': prob,
                'pred_evidence': evidences
            })
        
        # evaluate
        self.evaluate(pred_labels, target_labels, logger, 3)

        return results


def main(args):
    # init log
    log_path = args.log_path + args.dataset_name + "_execute_program_with_deberta.log"
    # log_path = args.log_path + "debug.log"
    logger = log.get_logger(log_path)
    logger.info(args)

    # load generated programs
    logger.info("loading generated programs......")
    program_path = args.program_path + args.dataset_name + "_" + args.mode + args.program_file
    with open(program_path, 'r') as f:
        program_data = json.load(f)
    
    dataset = program_data if args.num_eval_samples < 0 else program_data[:args.num_eval_samples]
    
    # load model
    logger.info("loading model......")
    program_execute = Program_Execute(args, logger)
    results = program_execute.execute_program(logger, dataset)

    # finish prediction
    logger.info("prediction finished...")
    out_path = args.output_path + args.dataset_name + "_" + args.mode + "_results.json"
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--raw_data_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='FEVER')
    parser.add_argument('--program_path', type=str, default='./results/programs/')
    parser.add_argument('--program_file', type=str, default='_N1_aquilacode-7b-nv_programs_with_evidence.json')
    # tfidf arguments
    parser.add_argument('--max_page', type=int, default=3)
    parser.add_argument('--max_sent', type=int, default=5)
    parser.add_argument('--tfidf_model', type=str, default='./data/Wikipedia/data/tfidf.npz')
    # wikipedia arguments
    parser.add_argument('--db_path', type=str, default='./data/Wikipedia/data/wikipedia.db')
    parser.add_argument('--db_table', type=str, default='documents')

    parser.add_argument('--log_path', type=str, default='./logs/')
    # model arguments
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev'])
    parser.add_argument('--output_path', type=str, default='./results/fact_checking/')
    parser.add_argument('--max_deberta_tokens', type=int, default=512)
    # parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    # parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
    parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    # training arguments
    parser.add_argument('--num_retrieved', type=int, default=5)
    parser.add_argument('--num_eval_samples', type=int, default=-1)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    main(args)
