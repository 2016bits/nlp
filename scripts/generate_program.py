import argparse
import torch
import os
import json
from tqdm import tqdm

from utils.data import RawData
from utils import log, flagai_model
from prompt import Prompt_Loader


class Program_Generate:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.num_programs_per_example = args.num_programs_per_example

        self.ai_model = flagai_model.FlagAIModel(args)
        self.prompt_loader = Prompt_Loader()

        self.result_dict = []
    
    def update_results(self, sample, generated_text):
        program_list = [operation.strip() for operation in generated_text.split('\n')[1:]]
        self.result_dict[sample['id']]['predicted_program'].append(program_list)
    
    def batch_generate_program(self, logger, datasets):
        self.result_dict = []
        # create output_dir
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # generate programs
        temperature = 0.0 if self.num_programs_per_example == 1 else 0.7
        outputs = []

        # initialize empty results
        result_dict = {}
        for idx, sample in enumerate(datasets):
            result = {
                'idx': idx,
                'id': sample['id'],
                'claim': sample['claim'],
                'label': sample['label'],
                'evidence': sample['evidence'],
                'predicted_program': []
            }
            result_dict[sample['id']] = result
        self.result_dict = result_dict

        # for each iteration
        for iteration in range(self.num_programs_per_example):
            logger.info("Generating programs for iteration {}...".format(iteration + 1))
            # for each batch
            for example in tqdm(datasets):
                # create prompt
                prompt = self.prompt_loader.prompt_construction(example['claim'], self.dataset_name)
                try:
                    output = self.ai_model.generate_program(prompt, temperature)
                    self.update_results(example, output)
                    logger.info("Successfully generate reasoing programs for example: {}".format(example['id']))
                except:
                    logger.info('Error in generating reasoning programs for example: {}'.format(example['id']))
        
        logger.info('Generated {} examples.'.format(len(result_dict)))
        # create outputs
        for key in result_dict:
            outputs.append(result_dict[key])
        sorted_outputs = sorted(outputs, key=lambda x: x['idx'])

        # save outputs
        save_path = self.save_path + '/' + self.dataset_name + '_' + self.args.mode + '_N' + str(self.num_programs_per_example) + '_' + self.model_name + '_programs_with_evidence.json'
        with open(save_path, 'w') as f:
            json.dump(sorted_outputs, f, indent=2, ensure_ascii=False)
        logger.info("Finished!")

def main(args):
    log_path = args.log_path + args.dataset_name + '_generate_program.log'

    # init logger
    logger = log.get_logger(log_path)
    logger.info(args)

    # load data
    logger.info("loading data......")
    data_path = args.data_path + args.dataset_name + "/processed/" + args.mode + ".json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    dataset = dataset if args.num_eval_samples < 0 else dataset[:args.num_eval_samples]

    # load model
    logger.info("loading model......")
    program_generate = Program_Generate(args)
    program_generate.batch_generate_program(logger, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='FEVER', choices=['SCIFACT'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--save_path', type=str, default='./results/programs')

    parser.add_argument('--num_eval_samples', type=int, default=-1)
    parser.add_argument('--num_programs_per_example', type=int, default=1)

    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=3)

    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--task_name', type=str, default='lm')
    parser.add_argument('--model_name', type=str, default='aquilacode-7b-nv')
    parser.add_argument('--use_cache', type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='../checkpoints_in/')
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_cuda:
        torch.cuda.set_device(int(args.gpu))
    
    main(args)
