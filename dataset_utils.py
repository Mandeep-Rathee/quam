
import torch
from torch.utils.data import DataLoader, Dataset


class MSMARCODataset(Dataset):
    def __init__(self, data, doc_store, tokenizer):
        self.data = data
        self.doc_store = doc_store
        self.tokenizer= tokenizer
        self.max_length = 512

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
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label':label
        }
    
