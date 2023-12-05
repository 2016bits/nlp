import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from utils import log
from select.data_loader import DataLoader
from select.models import inference_model

def correct_prediction(prob_pos, prob_neg):
    correct = 0.0
    prob_pos = prob_pos.view(-1).tolist()
    prob_neg = prob_neg.view(-1).tolist()
    assert len(prob_pos) == len(prob_neg)
    for step in range(len(prob_pos)):
        if prob_pos[step] > prob_neg[step]:
            correct += 1
    return correct

def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    for inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg in tqdm(validset_reader):
        prob_pos = model(inp_tensor_pos, msk_tensor_pos, seg_tensor_pos)
        prob_neg = model(inp_tensor_neg, msk_tensor_neg, seg_tensor_neg)
        correct_pred += correct_prediction(prob_pos, prob_neg)
    dev_accuracy = correct_pred / validset_reader.total_num
    return dev_accuracy

def main(args):
    log_path = args.log_path + args.dataset + "_train_{}_select_contrastive.log".format(args.train_batch_size)
    logger = log.get_logger(log_path)

    logger.info(args)
    logger.info('Start training!')

    # load data
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir, do_lower_case=False)
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, tokenizer, args, batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, tokenizer, args, batch_size=args.valid_batch_size)

    # load model
    logger.info('initializing estimator model')
    bert_model = BertModel.from_pretrained(args.cache_dir).to(args.device)
    model = inference_model(bert_model, args).to(args.device)
    
    save_path = args.outdir + '/bert_{}_'.format(args.train_batch_size)
    best_acc = 0.0
    running_loss = 0.0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_step = 0
    crit = nn.MarginRankingLoss(margin=1)

    # start training
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        for inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg in tqdm(trainset_reader):
            model.train()
            score_pos = model(inp_tensor_pos, msk_tensor_pos, seg_tensor_pos)
            score_neg = model(inp_tensor_neg, msk_tensor_neg, seg_tensor_neg)
            label = torch.ones(score_pos.size()).to(args.device)
            loss = crit(score_pos, score_neg, Variable(label, requires_grad=False))
            running_loss += loss.item()
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if global_step % 100 == 0:
                logger.info('Epoch: {}, Step: {}, Loss: {}'.format(epoch, global_step, (running_loss / global_step)))
            
        logger.info('Start eval!')
        eval_acc = eval_model(model, validset_reader)
        logger.info('Dev acc: {}'.format(eval_acc))
        if eval_acc >= best_acc:
            best_acc = eval_acc
            torch.save({'epoch': epoch,
                        'model': model.state_dict()}, save_path + "best.pt")
            logger.info("Saved best epoch {}, best acc {}".format(epoch, best_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--train_path', help='train path', default="./data/FEVER/SelectData/train_pair")
    parser.add_argument('--valid_path', help='valid path', default="./data/FEVER/SelectData/dev_pair")
    parser.add_argument('--log_path', type=str, default="./logs/")
    parser.add_argument('--dataset', type=str, default="FEVER")
    parser.add_argument('--outdir', help='path to output directory', default="./models")

    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--cache_dir', default="./bert-base-uncased")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:2")

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    main(args)
