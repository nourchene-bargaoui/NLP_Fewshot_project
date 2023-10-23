from transformers import BertModel
import torch
from torch import nn

class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_layer = nn.Linear(200, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Pass the output through the BiLSTM layer
        lstm_output, _ = self.bilstm(sequence_output)

        # Apply the linear layer to get NER label logits
        logits = self.linear_layer(lstm_output)

        return logits

class FewShotNERModel(nn.Module):
    def __init__(self, num_labels, num_classes):
        super(FewShotNERModel, self).__init__()
        self.ner_models = nn.ModuleList([NERModel(num_labels) for _ in range(num_classes)])

    def forward(self, class_idx, input_ids, attention_mask):

        # Forward pass through the selected NER model for the specified class
        ner_model = self.ner_models[class_idx]
        logits = ner_model(input_ids, attention_mask)
        return logits

# Usage example:
num_classes = 2  # Number of classes (N-way)
num_labels = 2  # Number of NER labels
few_shot_model = FewShotNERModel(num_labels, num_classes)


