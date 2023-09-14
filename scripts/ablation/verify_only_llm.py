import argparse
import torch
import os
import json
from tqdm import tqdm

from utils import log, flagai_model

class Verify_Only_LLM:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.num_programs_per_example = args.num_programs_per_example

        self.ai_model = flagai_model.FlagAIModel(args)

def main(args):
    log_path = args.log_path + args.dataset_name + '_verify_only_llm.log'

    # init logger
    logger = log.get_logger(log_path)
    logger.info(args)

    # load data
    logger.info("loading data......")
    data_path = args.data_path + args.dataset_name + "/" + args.mode + "_data.json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    dataset = dataset if args.num_eval_samples < 0 else dataset[:args.num_eval_samples]

    # load model
    logger.info("loading model......")
    model = Verify_Only_LLM(args)
    model.verify(logger, dataset)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='FEVER')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--save_path', type=str, default='./results/ablation/')

    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--num_eval_samples', type=int, default=-1)

    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--task_name', type=str, default='lm')
    parser.add_argument('--model_name', type=str, default='aquilacode-7b-nv')
    parser.add_argument('--use_cache', type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='../checkpoints_in/')
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
    
    main(args)
