from transformers import BertModel
import torch
from torch import nn

class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased") #bet
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True) #bilstn
        self.linear_layer = nn.Linear(200, num_labels)  # 100 * 2 (bidirectional) linear

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Pass the output through the BiLSTM layer
        lstm_output, _ = self.bilstm(sequence_output)

        # Apply the linear layer to get NER label logits
        logits = self.linear_layer(lstm_output)

        return logits
