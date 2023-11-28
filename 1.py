# Load model directly
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-large-msmarco-10k")
model = T5ForConditionalGeneration.from_pretrained("castorini/monot5-large-msmarco-10k").to("cuda:0")

# 准备要比较的两个句子
sentence1 = "This is the first sentence."
sentence2 = "This sentence is number one."

# 格式化输入为模型所需格式（输入和输出格式一致）
inputs = tokenizer.encode(f"stsb sentence1: {sentence1} sentence2: {sentence2}", return_tensors="pt").to("cuda:0")

# 使用MonT5模型计算相关性分数
with torch.no_grad():
    outputs = model.generate(inputs)

# 将输出解码为字符串，并移除特殊token
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
similarity_score = float(output_text)

print(f"Similarity score: {similarity_score}")
