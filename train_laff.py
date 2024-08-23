

import functools
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

import ir_datasets
import pyterrier as pt
pt.init()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_scheduler
import datasets

from base_models import BinaryClassificationT5Model, BinaryClassificationBertModel
from dataset_utils import MSMARCODataset 

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)




dataset = ir_datasets.load("msmarco-passage/train")
doc_store = dataset.docs_store()


run_ds = datasets.load_from_disk("data/laff_train_data")
data_list = []

def add_training_data(row):
    qid = row['qid']
    query = row['query']
    pos = row['pseudo_pos']
    neg = row['neg']
    s = row["S"]
    score = row['score']

    if len(s)!=0:
        for p in pos:
            for n in neg:
                data_list.append((p, n, s, score))     ## We create a tuple of positive, negative, S, and score


run_ds.map(add_training_data, desc="adding data points to data list")




base_model_name =  "bert-base-uncased" 
tokenizer = BertTokenizer.from_pretrained(base_model_name, torch_dtype=torch.float16)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1,torch_dtype=torch.float16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinaryClassificationBertModel(base_model)
model.to(device)


"""A custom dataset class is defined to load the data points from the data list."""
class MSMARCODatasetS(Dataset):
    def __init__(self, data, doc_store):
        self.data = data
        self.doc_store = doc_store

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        pos, neg, s, score, _ = data_item

        assert len(s) == len(score), f"Lengths of s and score are not the same."

        pos_text = [self.doc_store.get(item).text for item in pos]
        neg_text = [self.doc_store.get(item).text for item in neg]
        batch_s_text = [self.doc_store.get(s_id).text for s_id in s]

        return {
            "pos_text": pos_text,
            "neg_text": neg_text,
            "s_text": batch_s_text,
            "s": s,
            "score": score
        }


"""The function is used as the collate function for the DataLoader."""

def encode_batch(batch, tokenizer, max_length):
    texts = []
    s_texts = []
    s_list = []
    scores = []
    labels= []


    # Collect texts and scores
    for item in batch:
        pos_texts = [item['pos_text']] * len(item['s_text'])
        pos_label = [1]*len(item['s_text'])
        neg_texts = [item['neg_text']] * len(item['s_text'])
        neg_label = [0]*len(item['s_text'])
        texts.extend(pos_texts + neg_texts)
        labels.extend(pos_label+neg_label)        
        s_texts.extend(item['s_text']*2)
        s_list.extend([item['s']] * 2)  # Assuming 's' is the same for all s_texts in an item
        scores.extend([item['score']] * 2) # Assuming 'score' is the same for all s_texts in an item

    # Encode all texts in one call
    encodings = tokenizer.batch_encode_plus(
        list(zip(texts, s_texts)),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'token_type_ids': encodings['token_type_ids'],  # Assuming you still want token_type_ids
        's': s_list,
        'score': scores,
        'label':labels
    }



train_por = 0.95
batch_size= 16



train_dataset = MSMARCODatasetS(data_list[:int(train_por*len(data_list))],doc_store)
val_dataset = MSMARCODatasetS(data_list[int(train_por*len(data_list)):], doc_store)

custom_collate_fn = functools.partial(encode_batch, tokenizer=tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn )


optimizer = AdamW(model.parameters(), lr=0.0000003)  

bceloss = torch.nn.BCEWithLogitsLoss(reduction='mean')


def log_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f'Gradient norm for {name}: {grad_norm:.4f}')



num_epochs = 5

total_train_samples = len(train_dataset)
total_training_steps = (total_train_samples // batch_size) * num_epochs
num_warmup_steps = total_training_steps // 100 

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_training_steps
)


print(f"base_model={base_model_name}, S=top rank" )


if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        total_loss =0.0
        for batch in tqdm(train_loader):
            score = batch['score']

            input_ids= batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids=batch['token_type_ids'].to(device) 

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)
            label = torch.tensor(batch['label'], dtype=outputs.logits.dtype, device=device).unsqueeze(dim=-1)

            loss =  bceloss(outputs.logits, label)


            if torch.isnan(loss):
                print("NaN loss detected")
                continue
            
            print("per batch loss", loss.cpu().item())

            total_loss+=loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            scheduler.step()


        # Evaluation
        model.eval()
        val_preds = []
        val_true = []
        for batch in tqdm(val_loader):
            score = batch['score']
            input_ids= batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids=batch['token_type_ids'].to(device) 
            with torch.no_grad():
                
                outputs = model(input_ids=input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)

            prob = torch.nn.Sigmoid()(outputs.logits)
            threshold = 0.5
            predicted_labels = (prob >= threshold).float()
            val_preds.extend(predicted_labels.squeeze().cpu().numpy())
            val_true.extend(batch['label'])


        val_accuracy = accuracy_score(val_true, val_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy}')

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"laff-model/bert-base-laff.pth")    