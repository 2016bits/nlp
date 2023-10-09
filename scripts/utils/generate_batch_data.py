import torch
import sqlite3
from tqdm import tqdm

replacements = {
    "-LRB-": "(",
    "-LSB-": "[",
    "-LCB-": "{",
    "-RCB-": "}",
    "-RRB-": ")",
    "-RSB-": "]",
    "-COLON-": ":",
}

def convert_evidence(evidence_list, c):
    evidence_text = ""
    for title, sent_index in evidence_list:
        try:
            sql = """select * from {} where id = "{}" ;""".format("documents", title)
            cursor = c.execute(sql)
        except:
            sql = """select * from {} where id = '{}' ;""".format("documents", title)
            cursor = c.execute(sql)
        for row in cursor:
            lines = row[2].split('\n')
            for line in lines:
                try:
                    sent_id = eval(line.split('\t')[0])
                except:
                    continue
                if sent_id == sent_index:
                    sent_text = line.replace('{}\t'.format(sent_id), '')
                    sent_text = sent_text.replace('\t', ' ').replace('\n', ' ')
                    for replace in replacements:
                        if replace in sent_text:
                            sent_text = sent_text.replace(replace, replacements[replace])
                    evidence_text += " " + sent_text
    return evidence_text

class BatchData:
    def __init__(self, tokenizer, db_path, max_tokens, batch_size):
        self.db_path = db_path
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def construct_prompt(self, sample_list):
        # concate evidence and claim to construct prompt
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # for NEI, use other evidence in same batch
        evidence = ""

        max_len = 0
        raw_batch_data = []
        for sample in sample_list:
            if sample['label'] == 'SUPPORTS':
                evidence = convert_evidence(sample['evidence'], c)
                texts = f"{evidence}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "
                text_len = len(texts.split())
                if text_len > self.max_tokens:
                    max_len = text_len
                    evidence_len = self.max_tokens - len(sample['claim']) - 25
                    texts = f"{evidence[:evidence_len]}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "
                elif text_len > max_len:
                    max_len = text_len
                    evidence_len = self.max_tokens - len(sample['claim']) - 25
                    texts = f"{evidence[:evidence_len]}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "

                raw_batch_data.append({
                    'id': sample['id'],
                    'text': texts,
                    'label': 'true'
                })

            elif sample['label'] == 'REFUTES':
                evidence = convert_evidence(sample['evidence'], c)
                texts = f"{evidence}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "
                text_len = len(texts.split())
                if text_len > self.max_tokens:
                    max_len = text_len
                    evidence_len = self.max_tokens - len(sample['claim']) - 25
                    texts = f"{evidence[:evidence_len]}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "
                elif text_len > max_len:
                    max_len = text_len
                    evidence_len = self.max_tokens - len(sample['claim']) - 25
                    texts = f"{evidence[:evidence_len]}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "

                raw_batch_data.append({
                    'id': sample['id'],
                    'text': texts,
                    'label': 'false'
                })

            elif sample['label'] == 'NOT ENOUGH INFO':
                texts = f"{evidence}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "
                text_len = len(texts.split())
                if text_len > self.max_tokens:
                    max_len = text_len
                    evidence_len = self.max_tokens - len(sample['claim']) - 25
                    texts = f"{evidence[:evidence_len]}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "
                elif text_len > max_len:
                    max_len = text_len
                    evidence_len = self.max_tokens - len(sample['claim']) - 25
                    texts = f"{evidence[:evidence_len]}\nBased on the above information, is it true that {sample['claim']}? true, false or unknown? The answer is: "

                raw_batch_data.append({
                    'id': sample['id'],
                    'text': texts,
                    'label': 'unknown'
                })

        return {
            'raw_batch_data': raw_batch_data,
            'max_len': max_len + 5
        }
    
    def batch_data(self, one_batch_data):
        # pad
        raw_batch_data = one_batch_data['raw_batch_data']
        max_len = one_batch_data['max_len']

        index_list = []
        input_id_list = []
        mask_list = []
        label_list = []
        for data in raw_batch_data:
            encoded_text = self.tokenizer.encode_plus(
                text=data['text'],
                add_special_tokens=True,
                padding='max_length',
                max_length=max_len,
                truncation=True
            )
            label_ids = self.tokenizer.encode(
                text=data['label'],
                add_special_tokens=True
            )
            index_list.append(data['id'])
            input_id_list.append(encoded_text['input_ids'])
            mask_list.append(encoded_text['attention_mask'])
            label_list.append(label_ids)
        
        return {
            'index': torch.tensor(index_list),
            'input_ids': torch.tensor(input_id_list),
            'attention_mask': torch.tensor(mask_list),
            'labels': torch.tensor(label_list)
        }

    def batch_list_instances(self, dataset):
        # batch data
        data_num = len(dataset)
        total_batch_num = data_num // self.batch_size + 1 if data_num % self.batch_size != 0 else data_num // self.batch_size
        batched_data = []
        for index in tqdm(range(total_batch_num)):
            sample_list = dataset[index*self.batch_size: (index+1)*self.batch_size]
            one_batch_data = self.construct_prompt(sample_list)
            batched_data.append(self.batch_data(one_batch_data))
        return batched_data
