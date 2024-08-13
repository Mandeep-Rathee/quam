import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertModel


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
    """
    Binary Classification BERT Model.

    This class represents a binary classification model based on the BERT architecture.

    Args:
        base_model (nn.Module): The base BERT model.

    Attributes:
        base_model (nn.Module): The base BERT model.

    """

    def __init__(self, base_model):
        super(BinaryClassificationBertModel, self).__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (torch.Tensor): The token type IDs.

        Returns:
            outputs: model outputs.

        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs



    def dual_encoder(self, batch, device):
        """
        Dual encoder for binary classification.

        """
        pos_input_ids = batch['pos']['input_ids'].to(device)
        pos_attention_mask = batch['pos']['attention_mask'].to(device)
        pos_token_type_ids = batch['pos']['token_type_ids'].to(device)

        neg_input_ids = batch['neg']['input_ids'].to(device)
        neg_attention_mask = batch['neg']['attention_mask'].to(device)
        neg_token_type_ids = batch['neg']['token_type_ids'].to(device)

        pos_outputs = self.base_model(input_ids=pos_input_ids, attention_mask=pos_attention_mask, token_type_ids=pos_token_type_ids)
        neg_outputs = self.base_model(input_ids=neg_input_ids, attention_mask=neg_attention_mask, token_type_ids=neg_token_type_ids)

        return pos_outputs, neg_outputs
    


    


