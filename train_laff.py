

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


save_data = False   
train_data_type = "tct"
## code for converting the dataset to a list of tuples

if save_data:
    data_list = []

    ds_path = f"data/ms_marcopassage_train_50k_pseudo_pos_neg_nbh_Q_S_ds_{train_data_type}>>monoT5"
    run_ds = datasets.load_from_disk(ds_path)

# print(run_ds)
# print(run_ds[0])



def add_training_data(row):
    qid = row['qid']
    query = row['query']
    pos = row['pseudo_pos']
    neg = row['neg']
    s = row["S"]
    score = row['score']
    Q = row['Q']

    if len(s)!=0:
        for p in pos:
            for n in neg:
                data_list.append((p, n, s, score, Q))    


def add_training_data_S(row):
    pos = row['pseudo_pos']
    neg = row['neg']
    score = row['score']
    s = pos

    if len(pos)!=0:
        for p in pos:
            for n in neg:
                data_list.append((p, n, s, score))       


def add_training_data_S_ind(row):
    qid = row['qid']
    query = row['query']
    pos = row['pseudo_pos']
    neg = row['neg']

    if len(pos)!=0:
        data_list.append((pos, neg))


def add_all_data(row):
    qid = row['qid']
    query = row['query']
    pos = row['pseudo_pos']
    neg = row['neg']
    s = row["S"]
    score = row['score']
    Q = row['Q']

    data_list.append((pos, neg, s, score, Q))



if save_data:
    run_ds.map(add_all_data, desc="adding data points to data list")

    with open(f'data/ms_marcopassage_train_50k_pseudo_pos_neg_nbh_Q_S_ds_all_{train_data_type}>>monoT5.pkl', 'wb') as f:
        pickle.dump(data_list, f)



with open(f'data/ms_marcopassage_train_50k_pseudo_pos_neg_nbh_Q_S_ds_all_{train_data_type}>>monoT5.pkl', 'rb') as f:
    data_list = pickle.load(f)


# print(data_list[0])

# print(len(data_list))

# exit()

train_data = data_list



base_model_name =  "bert-base-uncased" #"bert_small"

if base_model_name!="bert-base-uncased":
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small", torch_dtype=torch.float16)
    base_model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-small", num_labels=1,torch_dtype=torch.float16)
else:
    tokenizer = BertTokenizer.from_pretrained(base_model_name, torch_dtype=torch.float16)
    base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1,torch_dtype=torch.float16)


#path = f"models/bert-base-uncased_ms_marcopassage_train_random_neg.pth"
path = f"models/bert-base-uncased_ms_marcopassage_train_50k_pseudo_pos_neg_S_ds_tct>>monoT5_epoch=5_loss=unbce.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BinaryClassificationBertModel(base_model)
# model.load_state_dict(torch.load(path, map_location=device))
model.to(device)


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

def encode_batchS(batch, tokenizer, max_length):
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


def encode_batch_bce(batch, tokenizer, max_length):
    texts = []
    s_texts = []
    s_list = []
    scores = []
    labels= []


    # Collect texts and scores
    for item in batch:
        assert len(item['s_text']) == len(item['pos_text']) == len(item['neg_text']), f"Lengths of s and score are not the same."

        for i in range(len(item['pos_text'])):
            pos_texts = [item['pos_text'][i]] * len(item['s_text'])
            neg_texts = [item['neg_text'][i]] * len(item['s_text'])
            texts.extend(pos_texts + neg_texts)    
            pos_label = [1]*len(pos_texts)
            neg_label = [0]*len(neg_texts)
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
#int(len(data_list)*train_por)
batch_size= 2 #128 #8



# train_dataset = MSMARCODatasetS(data_list[:1000], doc_store)
# val_dataset = MSMARCODatasetS(data_list[-40:], doc_store)

train_dataset = MSMARCODatasetS(data_list[:int(train_por*len(data_list))],doc_store)
val_dataset = MSMARCODatasetS(data_list[int(train_por*len(data_list)):], doc_store)

custom_collate_fn = functools.partial(encode_batch_bce, tokenizer=tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn )


optimizer = AdamW(model.parameters(), lr=0.0000003)  # Default learning rate is 0.0000003

bceloss = torch.nn.BCEWithLogitsLoss(reduction='mean')
bce = torch.nn.BCELoss()

#marginloss = torch.nn.MarginRankingLoss(margin=1.0)


def loss_fun(logits, score, margin=1.0):

    score = torch.tensor(score, device=device)
    score = F.softmax(score, dim=-1)

    logits = logits.reshape(score.size())
    aff_scores = torch.mul(score, logits).sum(dim=-1).reshape(-1,2) #ToDo add a scaling factor

    loss =  torch.clamp(margin - (torch.sigmoid(aff_scores[:,0].view(-1)) - torch.sigmoid(aff_scores[:,1].view(-1))), min=0).mean()

    return loss

margin=1.0

def margin_loss(pos_scores, neg_scores):
    return torch.clamp(margin - (torch.sigmoid(pos_scores.view(-1)) - torch.sigmoid(neg_scores.view(-1))), min=0).mean()


def bceloss_shuffle(pred,labels):
    num_samples = pred.size(0)
    perm_indices = torch.randperm(num_samples)
    shuffled_pred = pred[perm_indices]
    shuffled_true_labels = labels[perm_indices]

    return bceloss(pred, labels)


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

use_loss = "bce"

print(f"loss={use_loss}, base_model={base_model_name}, S=top rank" )

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


            # print("outputs", outputs.logits, outputs.logits.size())
            # print("label", label, label.size())
            # exit()

            if use_loss=="bce":
                loss =  bceloss_shuffle(outputs.logits, label)

            elif use_loss=="margin":
                loss = loss_fun(outputs.logits, score, 1.0)

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

        # if base_model_name=="bert-base-uncased":
        #     torch.save(model.state_dict(), f"models/{base_model_name}_ms_marcopassage_train_50k_curriculum_pseudo_rank_pos_neg_nbh_Q_S_ds_{train_data_type}>>monoT5_epoch={epoch+1}_loss={use_loss}.pth")
    
    #if base_model_name!="bert-base-uncased":
    torch.save(model.state_dict(), f"models/{base_model_name}_ms_marcopassage_train_50k_laff_curriculum_pseudo_pos=S_neg_ds_{train_data_type}>>monoT5_loss={use_loss}.pth")    