# Load model directly
import torch
import argparse
from torch.autograd import Variable
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
# from search_module import inference_model

def convert(tokenizer, sentence1, sentence2, max_len = 50):
    # 格式化输入为模型所需格式（输入和输出格式一致）
    tokens_a = tokenizer.tokenize(sentence1)
    tokens_b = tokenizer.tokenize(sentence2)
    tokens = ['[CLS]'] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    tokens += tokens_b + ["[SEP]"]
    segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_len - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, input_mask, segment_ids

def fun(args):
    tokenizer = BertTokenizer.from_pretrained(args.select_model_cache_dir, do_lower_case=False)
    bert_model = BertModel.from_pretrained(args.select_model_cache_dir).to(args.device)
    model = inference_model(bert_model, args).to(args.device)
    model.load_state_dict(torch.load(args.select_checkpoint)['model'])
    model.eval()

    # 准备要比较的两个句子
    sentence1 = "Saratoga is an American film from 1937."
    sentence2 = "Saratoga is a 1937 American romantic comedy film written by Anita Loos and directed by Jack Conway ."
    sentence3 = "The film stars Clark Gable and Jean Harlow in their sixth and final film collaboration , and features Lionel Barrymore , Frank Morgan , Walter Pidgeon , Hattie McDaniel , and Margaret Hamilton ."

    id1, mask1, seg1 = convert(tokenizer, sentence1, sentence2)
    id2, mask2, seg2 = convert(tokenizer, sentence1, sentence3)

    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    inp_padding.append(id1)
    msk_padding.append(mask1)
    seg_padding.append(seg1)
    inp_padding.append(id2)
    msk_padding.append(mask2)
    seg_padding.append(seg2)

    inp_tensor_input = Variable(
        torch.LongTensor(inp_padding)).to(args.device)
    msk_tensor_input = Variable(
        torch.LongTensor(msk_padding)).to(args.device)
    seg_tensor_input = Variable(
        torch.LongTensor(seg_padding)).to(args.device)

    probs = model(
        inp_tensor_input, msk_tensor_input, seg_tensor_input)

    print(f"Similarity score: {probs}")

def fun1(args):
    tokenizer = BertTokenizer.from_pretrained(args.select_model_cache_dir, do_lower_case=False)
    bert_model = BertForSequenceClassification.from_pretrained(args.select_model_cache_dir).to(args.device)
    model = inference_model(bert_model, args).to(args.device)

    sentence1 = "Saratoga is an American film from 1937."
    sentence2 = "Saratoga is a 1937 American romantic comedy film written by Anita Loos and directed by Jack Conway ."
    sentence3 = "The film stars Clark Gable and Jean Harlow in their sixth and final film collaboration , and features Lionel Barrymore , Frank Morgan , Walter Pidgeon , Hattie McDaniel , and Margaret Hamilton ."

    id1, mask1, seg1 = convert(tokenizer, sentence1, sentence2)
    id2, mask2, seg2 = convert(tokenizer, sentence1, sentence3)

    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    inp_padding.append(id1)
    msk_padding.append(mask1)
    seg_padding.append(seg1)
    inp_padding.append(id2)
    msk_padding.append(mask2)
    seg_padding.append(seg2)

    inp_tensor_input = Variable(
        torch.LongTensor(inp_padding)).to(args.device)
    msk_tensor_input = Variable(
        torch.LongTensor(msk_padding)).to(args.device)
    seg_tensor_input = Variable(
        torch.LongTensor(seg_padding)).to(args.device)

    probs = model(inp_tensor_input, msk_tensor_input, seg_tensor_input)
    print(probs)

def fun2(args):

    # 加载预训练的BERT模型和分词器
    model = BertModel.from_pretrained('./bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    # 输入文本
    text = "Saratoga is an American film from 1937."

    # 使用分词器对文本进行编码
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=20)
    import pdb
    pdb.set_trace()

    # 获取BERT模型的输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取CLS标记对应的隐藏状态作为句子向量表示
    sentence_vector = outputs.last_hidden_state[:, 0, :]  # [CLS] 标记对应的隐藏状态

    # 输出句子向量的维度
    print(sentence_vector.shape)  # 应该是 (1, hidden_size)，其中 hidden_size 是BERT模型的隐藏状态维度


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--select_model_cache_dir', default="./bert-base-uncased")
    parser.add_argument('--select_checkpoint', default="./models/bert.best.pt")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--device', type=str, default="cuda:2")

    args = parser.parse_args()
    fun2(args)
