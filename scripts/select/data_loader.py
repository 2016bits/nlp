import torch
import re
import json
import numpy as np
from torch.autograd import Variable

def text2tensor(text_list, tokenizer, max_len):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    for _, sample in enumerate(text_list):
        text1, text2 = sample
        text = "[CLS] {} [SEP] {}".format(text1, text2)
        tensor = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        inp_padding.append(tensor['input_ids'])
        msk_padding.append(tensor['attention_mask'])
        seg_padding.append(tensor['token_type_ids'])
    return torch.stack(inp_padding), torch.stack(msk_padding), torch.stack(seg_padding)

class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, batch_size=64):
        self.device = args.device

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.threshold = args.threshold
        self.data_path = data_path
        
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
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

    def read_file(self, data_path):
        examples = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                # fin: [claim, positive_title, positive_evidence, negative_title, negative_evidence]
                sublines = line.strip().split("\t")
                examples.append([self.process_sent(sublines[0]), self.process_sent(sublines[2]), self.process_sent(sublines[4])])
        return examples

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
            pos_inputs = list()
            neg_inputs = list()
            for example in examples:
                # example: [claim, positive_evidence, negative_evidence]
                pos_inputs.append([example[0], example[1]])
                neg_inputs.append([example[0], example[2]])
            
            inp_pos, msk_pos, seg_pos = text2tensor(pos_inputs, self.tokenizer, self.max_len)
            inp_neg, msk_neg, seg_neg = text2tensor(neg_inputs, self.tokenizer, self.max_len)

            inp_tensor_pos = Variable(torch.LongTensor(inp_pos)).to(self.device)
            msk_tensor_pos = Variable(torch.LongTensor(msk_pos)).to(self.device)
            seg_tensor_pos = Variable(torch.LongTensor(seg_pos)).to(self.device)
            inp_tensor_neg = Variable(torch.LongTensor(inp_neg)).to(self.device)
            msk_tensor_neg = Variable(torch.LongTensor(msk_neg)).to(self.device)
            seg_tensor_neg = Variable(torch.LongTensor(seg_neg)).to(self.device)

            self.step += 1
            return inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg
        else:
            self.step = 0
            self.shuffle()
            raise StopIteration()

class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, batch_size=64):
        self.device = args.device

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.threshold = args.threshold
        self.data_path = data_path
        inputs, ids, evi_list = self.read_file(data_path)
        self.inputs = inputs
        self.ids = ids
        self.evi_list = evi_list

        self.total_num = len(inputs)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
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

    def read_file(self, data_path):
        inputs = list()
        ids = list()
        evi_list = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                claim = instance['claim']
                id = instance['id']
                for evidence in instance['evidence']:
                    ids.append(id)
                    inputs.append([self.process_sent(claim), self.process_sent(evidence[2])])
                    evi_list.append(evidence)
        return inputs, ids, evi_list


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
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            ids = self.ids[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            evi_list = self.evi_list[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            
            inp_pos, msk_pos, seg_pos = text2tensor(inputs, self.tokenizer, self.max_len)

            inp_tensor_pos = Variable(torch.LongTensor(inp_pos)).to(self.device)
            msk_tensor_pos = Variable(torch.LongTensor(msk_pos)).to(self.device)
            seg_tensor_pos = Variable(torch.LongTensor(seg_pos)).to(self.device)

            self.step += 1
            return inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, ids, evi_list
        else:
            self.step = 0
            raise StopIteration()