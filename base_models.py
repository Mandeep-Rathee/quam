import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

# # Initialize tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained("t5-base")
# model = T5ForConditionalGeneration.from_pretrained("t5-base")

# # Freeze pre-trained parameters
# for param in model.parameters():
#     param.requires_grad = False

# Define binary classification head
class BinaryClassificationT5Model(nn.Module):
    def __init__(self, base_model):
        super(BinaryClassificationT5Model, self).__init__()
        self.base_model = base_model
        self.classification_head = nn.Linear(base_model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over sequence length
        logits = self.classification_head(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities



# Define binary classification model
class BinaryClassificationBertModel(nn.Module):
    def __init__(self, base_model):
        super(BinaryClassificationBertModel, self).__init__()
        self.base_model = base_model
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    


    


# base_model_name = "bert-base-uncased"

# tokenizer = BertTokenizer.from_pretrained(base_model_name) 
# base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=2)  # Binary classification
# base_model_2 = BertModel.from_pretrained(base_model_name)
# print(base_model)
# print("*"*150)

# print(base_model_2)