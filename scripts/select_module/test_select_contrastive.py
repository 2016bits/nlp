import argparse
import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# import select
from select_module.models import inference_model
from utils import log
from select_module.data_loader import DataLoaderTest


def save_to_file(results, out_path, topk):
    sorted_results = []
    for key, value in results.items():
        claim = value['claim']
        label = value['label']
        gold_evidence = value['gold_evidence']
        sorted_pred = sorted(value['pred_evidence'], key=lambda x:x[-1], reverse=True)
        sorted_results.append({
            'id': key,
            'claim': claim,
            'gold_label': label,
            'gold_evidence': gold_evidence,
            'pred_evidence': sorted_pred[:topk]
        })
    with open(out_path, 'w') as f:
        f.write(json.dumps(sorted_results, indent=2))

def eval_model(model, validset_reader):
    model.eval()
    results = {}
    for inp_tensor, msk_tensor, seg_tensor, ids, claims, labels, gold_evidences, pred_evidences in tqdm(validset_reader):
        probs = model(inp_tensor, msk_tensor, seg_tensor)
        probs = probs.tolist()
        assert len(probs) == len(ids) and len(probs) == len(claims) and len(probs) == len(gold_evidences) and len(probs) == len(pred_evidences)
        for i in range(len(probs)):
            if ids[i] not in results:
                results[ids[i]] = {
                    'claim': claims[i],
                    'label': labels[i],
                    'gold_evidence': gold_evidences[i],
                    'pred_evidence': []
                }
            #if probs[i][1] >= probs[i][0]:
            results[ids[i]]['pred_evidence'].append(pred_evidences[i] + [probs[i]])
    return results

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
    save_to_file(predict_dict, save_path, args.evi_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default="./results/select/test_data.json")
    parser.add_argument('--outdir', type=str, default="./results/select/", help='path to output directory')
    parser.add_argument('--log_path', type=str, default="./logs/")
    parser.add_argument('--dataset', type=str, default="FEVER")
    
    parser.add_argument("--batch_size", default=2048, type=int, help="Total batch size for training.")
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
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--device', type=str, default="cuda:1")

    args = parser.parse_args()
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    main(args)
