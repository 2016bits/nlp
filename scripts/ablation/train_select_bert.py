import os
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel

from utils.selectdata_loader import DataLoader

from utils import log

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    if 1.0 - x >= 0.0:
        return 1.0 - x
    return 0.0

def correct_prediction(prob_pos, prob_neg):
    correct = 0.0
    prob_pos = prob_pos.view(-1).tolist()
    prob_neg = prob_neg.view(-1).tolist()
    assert len(prob_pos) == len(prob_neg)
    for step in range(len(prob_pos)):
        if prob_pos[step] > prob_neg[step]:
            correct += 1
    return correct

class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        #self.proj_hidden = nn.Linear(self.bert_hidden_dim, 128)
        self.proj_match = nn.Linear(self.bert_hidden_dim, 1)


    def forward(self, inp_tensor, msk_tensor, seg_tensor):
        inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor).last_hidden_state
        inputs = self.dropout(inputs)
        score = self.proj_match(inputs).squeeze(-1)
        score = torch.tanh(score)
        return score

def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    for inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg in validset_reader:
        prob_pos = model(inp_tensor_pos, msk_tensor_pos, seg_tensor_pos)
        prob_neg = model(inp_tensor_neg, msk_tensor_neg, seg_tensor_neg)
        correct_pred += correct_prediction(prob_pos, prob_neg)
    dev_accuracy = correct_pred / validset_reader.total_num
    return dev_accuracy

def train_model(model, args, trainset_reader, validset_reader, logger):
    save_path = args.outdir + '/bert'
    best_acc = 0.0
    running_loss = 0.0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_step = 0
    crit = nn.MarginRankingLoss(margin=1)
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
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                eval_acc = eval_model(model, validset_reader)
                logger.info('Dev acc: {0}'.format(eval_acc))
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    torch.save({'epoch': epoch,
                                'model': model.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--train_path', help='train path', default="./data/FEVER/SelectData/dev_pair")
    parser.add_argument('--valid_path', help='valid path', default="./data/FEVER/SelectData/train_pair")
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

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    log_path = args.log_path + args.dataset + "_train_select_bert.log"
    logger = log.get_logger(log_path)
    logger.info(args)
    logger.info('Start training!')

    tokenizer = BertTokenizer.from_pretrained(args.cache_dir, do_lower_case=False)
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, tokenizer, args, batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, tokenizer, args, batch_size=args.valid_batch_size, test=True)

    logger.info('initializing estimator model')
    bert_model = BertModel.from_pretrained(args.cache_dir)
    bert_model = bert_model.to(args.device)
    model = inference_model(bert_model, args)
    model = model.to(args.device)
    train_model(model, args, trainset_reader, validset_reader, logger)
