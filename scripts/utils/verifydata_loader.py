import json
import re
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

label_dict = {
    'SUPPORTS': 0,
    'NOT ENOUGH INFO': 1,
    'REFUTES': 2
}



class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, tokenizer, args, dataset, batch_size=64):
        self.device = args.device

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        examples, labels = self.read_file(dataset)
        self.examples = examples
        self.labels = labels
        self.total_num = len(self.examples)
        self.total_step = self.total_num / batch_size
        self.shuffle()
        self.step = 0

    def process_sent(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title

    def convert_evidence(self, evidence_list):
        text = ""
        if not evidence_list:
            return text
        for evidence in evidence_list:
            if evidence and len(evidence) > 2 and evidence[0] and evidence[2]:
                text += "<title> {} <sentence> {} ".format(self.process_wiki_title(evidence[0]), self.process_sent(evidence[2]))
        return text

    def read_file(self, dataset):
        """读取数据，返回text列表和对应的label列表"""
        examples = []
        labels = []
        # 样例组成：claim+pred_evidence，claim+gold_evidence
        for inst in tqdm(dataset):
            claim = self.process_sent(inst['claim'])
            label = label_dict[inst['gold_label']]

            # claim+gold_evidence
            if label == 'NOT ENOUGH INFO':
                text = "<claim> {} <evidence>".format(claim)
                examples.append(text)
                labels.append(label)
            else:
                for evidence in inst['gold_evidence']:
                    text = "<claim> {} <evidence> {}".format(claim, self.convert_evidence(evidence))
                    examples.append(text)
                    labels.append(label)
            
            # claim+gold_evidence
            text = "<claim> {} <evidence> {}".format(claim, self.convert_evidence(inst['pred_evidence']))
            examples.append(text)
            labels.append(label)
        
        return examples, labels

    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            examples = self.examples[self.step * self.batch_size : (self.step+1)*self.batch_size]
            labels = self.labels[self.step * self.batch_size : (self.step+1)*self.batch_size]
            
            ids, msks = [], []
            for example in examples:
                tensors = self.tokenizer.encode_plus(example, truncation=True, padding='max_length', max_length=self.max_len)
                ids.append(tensors['input_ids'])
                msks.append(tensors['attention_mask'])
            id_tensors = torch.LongTensor(ids).to(self.device)
            msk_tensors = torch.LongTensor(msks).to(self.device)
            label_ids = torch.LongTensor(labels).to(self.device)
            self.step += 1
            return id_tensors, msk_tensors, label_ids
        else:
            self.step = 0
            self.shuffle()
            raise StopIteration()