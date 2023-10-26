import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
cache_dir = "./bert-base-uncased"
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)

sentence1 = "Trump won the 2020 election."
sentence2 = "Trump lost the 2020 election."

# 定义计算相似度的函数
def calc_similarity(s1, s2):
    # 对句子进行分词，并添加特殊标记
    encoded_sent1 = tokenizer.encode(sentence1, add_special_tokens=True)
    encoded_sent2 = tokenizer.encode(sentence2, add_special_tokens=True)

    # 获取句子向量 
    sent1_vec = model(torch.tensor([encoded_sent1]).to(device))[0][:,0,:]
    sent2_vec = model(torch.tensor([encoded_sent2]).to(device))[0][:,0,:]

    # 计算相似度
    cosine_sim = torch.nn.functional.cosine_similarity(sent1_vec, sent2_vec)
    return cosine_sim


print("Similarity score:", calc_similarity(sentence1, sentence2))