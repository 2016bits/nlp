import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class RawData:
    def __init__(self, id, claim, label):
        self.id = id
        self.claim = claim
        self.label = label


def merge_dataset(raw_data, program_data):
    evidence_data = {data['id']: data['evidence'] for data in raw_data}
    dataset = []
    for data in program_data:
        if data['id'] in evidence_data:
            new_data = {
                'idx': data['idx'],
                'id': data['id'],
                'claim': data['claim'],
                'label': data['label'],
                'predicted_program': data['predicted_program'],
                'evidence': evidence_data[data['id']]
            }
            if new_data not in dataset:
                dataset.append(new_data)
    return dataset


class MyDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        """return the length of dataset"""
        return len(self.dataset)
    
    def __getitem__(self, index):
        text_ids = self.tokenizer.encode_plus(
            text=self.dataset[index]['text'],
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len
        )
        label_ids = self.tokenizer.encode(
            text=self.dataset[index]['label'],
            add_special_tokens=True
        )
        return {
            "input_ids": text_ids['input_ids'],
            "attention_mask": text_ids['attention_mask'],
            "labels": label_ids
        }
    
    def collate_fn(self, batch):
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        labels_list = [torch.tensor(instance['labels']) for instance in batch]
        labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "labels": torch.tensor(labels_pad)
        }