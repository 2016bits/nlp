import torch
import torch.nn as nn

class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.pred_model = bert_model
        self.proj_match = nn.Linear(self.bert_hidden_dim, 1)

    def forward(self, input_ids, query_mask, query_seg):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        query_mask = query_mask.view(-1, query_mask.size(-1))
        query_seg = query_seg.view(-1, query_seg.size(-1))
        with torch.no_grad():
            outputs = self.pred_model(input_ids, attention_mask=query_mask, token_type_ids=query_seg)

        # 获取CLS标记对应的隐藏状态作为句子向量表示
        sentence_vector = outputs.last_hidden_state[:, 0, :].view(batch_size, seq_len, -1)
        dropout_vertor = self.dropout(sentence_vector)
        proj_vector = self.proj_match(dropout_vertor).squeeze(-1)
        score = torch.tanh(proj_vector)
        return score
    