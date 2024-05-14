### Input: train data with (q, d+) pairs
### Output: Model f, which will predict the score between (q, d+) and (q, d-)

### Steps: Make batches of (q,d+) and (q,d-)

from itertools import combinations
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

import ir_datasets
from pyterrier_adaptive import GAR, CorpusGraph
from pyterrier_pisa import PisaIndex
import pyterrier as pt
pt.init()

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import BertTokenizer, BertForSequenceClassification


from base_models import BinaryClassificationT5Model, BinaryClassificationBertModel

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)


retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
with open('data/ms_marcopassage_train_random_neg.pkl', 'rb') as f:
    data_list = pickle.load(f)


#graph = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_bm25_k16').to_limit_k(16)
#graph = CorpusGraph.load("msmarco-passage.gtcthnp.1024").to_limit_k(10)

dataset = ir_datasets.load("msmarco-passage/train")

doc_store = dataset.docs_store()

doc_dict = doc_store.get_many(['0', '100'])
text_list = [doc.text for doc in doc_dict.values()]   
query_id_to_text = {query.query_id: query.text for query in dataset.queries_iter()}


warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy.*")

class MSMARCODataset(Dataset):
    def __init__(self, data, doc_store, tokenizer,max_length):
        self.data = data
        self.doc_store = doc_store
        self.tokenizer= tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        passage_id_s, passage_id_t, label = data_item 
        passage_id_s_text = self.doc_store.get(passage_id_s).text
        passage_id_t_text = self.doc_store.get(passage_id_t).text

        encoding = self.tokenizer.encode_plus(
            passage_id_s_text,
            passage_id_t_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )


        return  {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label':label
        }

# tokenizer = T5Tokenizer.from_pretrained('t5-base')
# base_model = T5ForConditionalGeneration.from_pretrained('t5-base')

base_model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(base_model_name)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1)  # Binary classification

model = BinaryClassificationBertModel(base_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


train_por = 0.95

train_data = data_list[:int(train_por*len(data_list))]
val_data = data_list[int(train_por*len(data_list)):]

train_dataset = MSMARCODataset(train_data, doc_store, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = MSMARCODataset(val_data, doc_store, tokenizer, max_length=512)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# for data in train_loader:
#     print(data["label"], data["label"].shape, data["label"].squeeze().shape)

# exit()


optimizer = AdamW(model.parameters(), lr=0.00003)
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss =0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().unsqueeze(dim=-1).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # print("logits", logits, logits.shape)
        # prob = torch.nn.Sigmoid()(logits)
        # print("prob", prob, prob.shape)
        # print("labels", labels, labels.shape)


        loss = loss_fn(logits, labels)

        print("per batch loss", loss.cpu().item())

        total_loss+=loss
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    val_preds = []
    val_true = []
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            prob = torch.nn.Sigmoid()(logits)
            threshold = 0.5
            predicted_labels = (prob >= threshold).float()


    
        val_preds.extend(predicted_labels.squeeze().cpu().numpy())
        val_true.extend(batch['label'].squeeze().cpu().numpy())

    val_accuracy = accuracy_score(val_true, val_preds)
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy}')


torch.save(model.state_dict(), f"models/{base_model_name}_ms_marcopassage_train_random_neg.pth")