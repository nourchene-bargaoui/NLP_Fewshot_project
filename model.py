from transformers import BertModel
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
    def forward(self, ids, attention_mask):
        bert_sequences = self.bert(ids, attention_mask=attention_mask)
        #print(bert_sequences)
