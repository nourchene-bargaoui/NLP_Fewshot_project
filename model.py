from transformers import BertModel
import torch
from torch import nn

class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_layer = nn.Linear(200, num_labels)  # 100 * 2 (bidirectional)
        self.device = "cuda:0"

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        outputs.to(self.device)
        sequence_output = outputs.last_hidden_state
        sequence_output.to(self.device)
        sequence_output = self.dropout(sequence_output)
        sequence_output.to(self.device)

        # Pass the output through the BiLSTM layer
        lstm_output, _ = self.bilstm(sequence_output)
        lstm_output.to(self.device)

        # Apply the linear layer to get NER label logits
        logits = self.linear_layer(lstm_output)
        logits.to(self.device)

        return logits
