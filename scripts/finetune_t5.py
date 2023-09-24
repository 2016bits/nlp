import sys
import torch
import json
import argparse
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from utils import log
from utils.process_data import preprocess_dataset
from utils.data import MyDataset

label_map = {
    "true": 0,
    "false": 1,
    "uninformed": 2
}

def validate(model, device, data_loader, label_id_list, epoch, logger, tokenizer):
    model.eval()

    predicted_labels = torch.LongTensor([]).to(device)
    target_labels = torch.LongTensor([]).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for _, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
        pred_labels = outputs[:, 1]

        target_labels = torch.cat([target_labels, labels[:, 0]])
        predicted_labels = torch.cat([predicted_labels, pred_labels])

        for pred in pred_labels:
            if pred not in label_id_list:
                text = tokenizer.batch_decode(torch.tensor([pred]), skip_special_tokens=True)
                logger.info(f"The predicted label is not in label_id_list, its index is {text}")

        accuracy = accuracy_score(target_labels.tolist(), predicted_labels.tolist())
        macro_f1 = f1_score(target_labels.tolist(), predicted_labels.tolist(), average='macro')
        micro_f1 = f1_score(target_labels.tolist(), predicted_labels.tolist(), average='micro')
        weighted_f1 = f1_score(target_labels.tolist(), predicted_labels.tolist(), average='weighted')

        data_loader.desc = "[valid epoch {}] acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
            epoch, accuracy, macro_f1, micro_f1, weighted_f1
        )
        logger.info("[valid epoch {}] acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
            epoch, accuracy, macro_f1, micro_f1, weighted_f1
        ))

    return accuracy

def main(args):
    log_path = args.log_path + args.dataset + "_finetune_t5.log"
    logger = log.get_logger(log_path)
    logger.info(args)

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # load data
    logger.info("loading data......")
    train_data_path = args.data_path + args.dataset + "/processed/train.json"
    dev_data_path = args.data_path + args.dataset + "/processed/dev.json"
    with open(train_data_path, 'r') as f:
        train_dataset = json.load(f)
    train_data, max_len = preprocess_dataset(train_dataset, args.db_path, args.shot)
    train_set = MyDataset(train_data, tokenizer, max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn)

    # encode label ids
    label_id_list = []
    for label in label_map.keys():
        label_ids = tokenizer.encode(text=label, add_special_tokens=False)
        label_id_list.append(label_ids)

    with open(dev_data_path, 'r') as f:
        dev_dataset = json.load(f)
    dev_data, max_len = preprocess_dataset(dev_dataset, args.db_path, args.shot)
    dev_set = MyDataset(dev_data, tokenizer, max_len)
    dev_loader = DataLoader(dev_set, batch_size=1, collate_fn=dev_set.collate_fn)
    
    # load model
    logger.info("loading model......")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(args.device)

    if args.use_Adafactor and args.use_AdafactorSchedule:
        # https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/optimizer_schedules#transformers.Adafactor
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
    elif args.use_Adafactor and not args.use_AdafactorSchedule:
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=args.lr_warmup_steps,
                                                       num_training_steps=len(train_loader) * args.epoch)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=args.lr_warmup_steps,
                                                       num_training_steps=len(train_loader) * args.epoch)

    # frozen parameters of encoder
    for param in model.get_encoder().parameters():
        param.requires_grad = False
    
    best_macro_f1 = 0
    for epoch in range(args.epoch):
        model.train()

        predicted_labels = torch.LongTensor([]).to(args.device)
        target_labels = torch.LongTensor([]).to(args.device)

        sum_loss = torch.zeros(1).to(args.device)
        optimizer.zero_grad()

        data_loader = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            labels = data['labels'].to(args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            pred_labels = torch.max(logits, dim=-1).indices

            predicted_labels = torch.cat([predicted_labels, pred_labels[:, 0]])
            target_labels = torch.cat([target_labels, labels[:, 0]])

            accuracy = accuracy_score(target_labels.tolist(), predicted_labels.tolist())

            loss.backward()

            sum_loss += loss.detach()
            avg_loss = sum_loss.item() / (step + 1)

            data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, acc: {:.3f}".format(
                epoch, optimizer.param_groups[0]["lr"], avg_loss, accuracy
            )
            logger.info("[train epoch {}] lr: {:.5f}, loss: {:.3f}, acc: {:.3f}".format(
                epoch, optimizer.param_groups[0]["lr"], avg_loss, accuracy
            ))

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        
        dev_macro_f1 = validate(model, args.device, dev_loader, label_id_list, epoch, logger, tokenizer)
        if dev_macro_f1 > best_macro_f1:
            logger.info("save model at epoch {}, macro_f1: {}".format(epoch, dev_macro_f1))
            if args.shot == -1:
                shot_num = "_all_train_data"
            else:
                shot_num = "_{}shot_train_data".format(args.shot)
            save_model_path = args.save_model_path + args.dataset + shot_num + '.pth'
            torch.save(model.state_dict(), save_model_path)
            best_macro_f1 = dev_macro_f1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--dataset', type=str, default="FEVER")
    parser.add_argument('--data_path', type=str, default='./data/')

    parser.add_argument('--db_path', type=str, default='./data/Wikipedia/data/wikipedia.db')
    parser.add_argument('--model_name', type=str, default='google/flan-t5-xl')
    parser.add_argument('--cache_dir', type=str, default='./google/flan-t5-xl')
    parser.add_argument('--save_model_path', type=str, default='./model/finetuned_t5_')

    parser.add_argument('--use_Adafactor', type=bool, default=True)
    parser.add_argument('--use_AdafactorSchedule', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_warmup_steps', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:2")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--way', type=int, default=3)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=10)

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    main(args)
