import torch
import argparse
import json
import re
from tqdm import tqdm

from utils import log

class Program_Parser:
    def __init__(self):
        self.results = []
    
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
    def parse_search_command(self, command):
        # after executing search command, get variable
        return_var, tmp = command.split('= Search_pages')
        return_var = return_var.strip()

        # get parameter of search function
        matching = re.search(r'Search_pages\((.*?)\)', command)
        if matching:
            arguments = matching.group(1).split(', ')
            texts = [arg.strip('"') for arg in arguments]
        else:
            texts = tmp

        return texts

    # parse find command to get variable and parameters
    def parse_find_command(self, command):
        return_var, tmp = command.split('= Find_evidences')
        return_var = return_var.strip()

        pattern = r'Find_evidences\((.+?),(.+)\)'
        matching = re.search(pattern, command)
        param1 = matching.group(1)
        param2 = matching.group(2)
        params = re.findall(r'"(.+?)"', param2)
        params = list(set(params))
        key_words = params if len(params) > 0 else tmp

        return key_words

    # parse program
    def parse_program(self, program):
        claim = None
        document = None
        sentence = None
        
        # parse command line by line
        for command in program:
            cmd_type = self.get_command_type(command)

            # definite variable
            if cmd_type == "DEFINITE":
                claim = self.parse_definite_command(command)

            # search relevant wikipedia pages
            elif cmd_type == "SEARCH":
                # complete ) and "
                quote_num = len(re.findall(r'"', command))
                if quote_num % 2:
                    command += '"'
                if command[-1] != ')':
                    command += ')'

                document = self.parse_search_command(command)
                
            # find relevant evidences
            elif cmd_type == "FIND":
                # complete ) and "
                quote_num = len(re.findall(r'"', command))
                if quote_num % 2:
                    command += '"'
                if command[-1] != ')':
                    command += ')'

                key_words = self.parse_find_command(command)
                sentence = key_words
                
        # TODO: verify the claim

        return document, sentence
    
    def parse(self, logger, dataset):
        logger.info("start parsing programs...")

        for inst in tqdm(dataset):
            for program in inst['predicted_program']:
                document, sentence = self.parse_program(program)
                self.results.append({
                    'id': inst['id'],
                    'claim': inst['claim'],
                    'gold_label': inst['label'],
                    'gold_evidence': inst['evidence'],
                    'parsed_title': document,
                    'parsed_sentence': sentence
                })
        return self.results


def main(args):
    # init log
    log_path = args.log_path + args.dataset + "_search_on_programs.log"
    # log_path = args.log_path + "debug.log"
    logger = log.get_logger(log_path)
    logger.info(args)

    # load generated programs
    logger.info("loading generated programs......")
    program_path = args.program_path + args.dataset + "_" + args.mode + args.program_file
    with open(program_path, 'r') as f:
        dataset = json.load(f)
    
    # load model
    logger.info("loading model......")
    program_parser = Program_Parser()
    results = program_parser.parse(logger, dataset)

    # finish parsing
    logger.info("Parser finished...")
    out_path = args.output_path + args.dataset + "_" + args.mode + "_parse_results.json"
    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
     # data arguments
    parser.add_argument('--program_path', type=str, default='./results/programs/')
    parser.add_argument('--program_file', type=str, default='_N1_aquilacode-7b-nv_programs_with_evidence_new.json')
    parser.add_argument('--output_path', type=str, default='./results/parse/')
    parser.add_argument('--dataset', type=str, default="FEVER")
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev'])

    args = parser.parse_args()

    main(args)
