import os
import argparse
import torch
import json
from tqdm import tqdm
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel

from select.models import inference_model
from utils import log
from select.data_loader import DataLoaderTest


def save_to_file(all_predict, outpath):
    with open(outpath, "w") as out:
        for key, values in all_predict.items():
            sorted_values = sorted(values, key=lambda x:x[-1], reverse=True)
            data = json.dumps({"id": key, "evidence": sorted_values[:5]})
            out.write(data + "\n")

def eval_model(model, validset_reader):
    model.eval()
    all_predict = dict()
    for inp_tensor, msk_tensor, seg_tensor, ids, evi_list in validset_reader:
        probs = model(inp_tensor, msk_tensor, seg_tensor)
        probs = probs.tolist()
        assert len(probs) == len(evi_list)
        for i in range(len(probs)):
            if ids[i] not in all_predict:
                all_predict[ids[i]] = []
            #if probs[i][1] >= probs[i][0]:
            all_predict[ids[i]].append(evi_list[i] + [probs[i]])
    return all_predict

def main(args):
    log_path = args.log_path + args.dataset + "_test_select_contrastive.log"
    logger = log.get_logger(log_path)
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    logger.info("loading data...")
    validset_reader = DataLoaderTest(args.test_path, tokenizer, args, batch_size=args.batch_size)

    logger.info('loading model...')
    bert_model = BertModel.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.to(args.device)
    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)['model'])
    model = model.to(args.device)
    logger.info('Start eval!')
    save_path = args.outdir + "select_bert_contrastive.json"
    predict_dict = eval_model(model, validset_reader)
    save_to_file(predict_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default="./data/FEVER/processed/test.json")
    parser.add_argument('--outdir', type=str, default="./results/select/", help='path to output directory')
    parser.add_argument('--log_path', type=str, default="./logs/")
    parser.add_argument('--dataset', type=str, default="FEVER")
    
    parser.add_argument("--batch_size", default=4096, type=int, help="Total batch size for training.")
    parser.add_argument('--bert_pretrain', type=str, default="./bert-base-uncased")
    parser.add_argument('--checkpoint', type=str, default="./models/bert_4096_best.pt")
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda:1")

    args = parser.parse_args()
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    main(args)
