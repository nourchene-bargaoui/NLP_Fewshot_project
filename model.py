from transformers import BertModel
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear_layer = nn.Linear(in_features=768, out_features=26)

    def forward(self, ids, attention_mask):
        bert_sequences = self.bert(ids, attention_mask=attention_mask)
        pooled_output = bert_sequences.pooler_output
        linear_output = self.linear_layer(pooled_output)
        return linear_output