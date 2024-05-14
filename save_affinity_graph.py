import ir_datasets
import numpy as np
import json

import tempfile
import more_itertools
import datasets

from npids import Lookup
from pathlib import Path
import torch
import warnings
from tqdm import tqdm
from lz4.frame import LZ4FrameFile
logger = ir_datasets.log.easy()


import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from base_models import  BinaryClassificationBertModel

from dataset_utils import *

warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy.*")


out_dir_existing = "msmarco-passage.gbm25.1024"
out_dir_new = "msmarco-passage.gbm25.affinity.128"
out_dir_existing = Path(out_dir_existing)
out_dir_new = Path(out_dir_new)

edges_path_new = out_dir_new/'edges.u32.np'
edges_path_existing = out_dir_existing/'edges.u32.np'


weights_path_new = out_dir_new/'weights.f16.np'

edge_weights = [] ## from hugging face datasets
doc = [] ### huggingface from datasets


# Create a lookup object

docnos = Lookup(out_dir_existing/'docnos.npids')
print(len(docnos))
print(docnos[1])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 128

def edges_data(edges_path_):
    res = np.memmap(edges_path_, mode='r', dtype=np.uint32).reshape(-1, k)
    res = res[:, :k]
    return res

edges_data_store = edges_data(edges_path_existing )

print("edges data loaded")

def neighbours(doc_id):
    as_str = isinstance(doc_id, str)
    if as_str:
      doc_id = docnos.inv[doc_id]
    neigh = edges_data_store[doc_id]
    if as_str:
      neigh = docnos.fwd[neigh]

    return neigh



base_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(base_model_name)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1, torch_dtype=torch.float16)
model = BinaryClassificationBertModel(base_model)
model.load_state_dict(torch.load(f"models/{base_model_name}_ms_marcopassage_train_random_neg.pth"))
model.to(device)


dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

def affinity_scores(row):
    chunk = row["docno"]
    nbatch = len(chunk)
    dids = []
    sim_scores = []
    data_list = []

    for doc in chunk:
        for n in neighbours(doc):
            dids.append(n)
            data_list.append((doc, n, 1))

    assert len(data_list)==len(dids)==k*len(chunk) 

    batch_dataset = MSMARCODataset(data_list, docstore,  tokenizer, max_length=512)
    loader = DataLoader(batch_dataset, batch_size=nbatch*k)


    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Convert logits to probabilities
        #redicted_labels = torch.argmax(logits, dim=1) 
        scores = [tensor_item.item() for tensor_item in logits] 

        sim_scores+=scores

    assert len(data_list)==len(dids)==k*len(chunk) ==len(sim_scores)


    separated_dids = []
    separated_scores = []


    for i in range(len(chunk)):
        start_index = i * k
        end_index = start_index + k
        
        batch_docid = dids[start_index:end_index]
        batch_scores = scores[start_index:end_index]
        
        separated_dids.append(batch_docid)
        separated_scores.append(batch_scores)


    return {"dids":separated_dids, "scores": separated_scores}


# def from_corpus_to_affinity(affinity_path, edges_path, weights_path, batch_size, k):        
#         with ir_datasets.util.finialized_file(str(edges_path), 'wb') as fe, ir_datasets.util.finialized_file(str(weights_path), 'wb') as fw: 
#             for chunk in more_itertools.chunked(logger.pbar(docnos, miniters=1, smoothing=0, desc='generating affinity scores', total=len(docnos)), batch_size):
#                     res_dids, res_scores = affinity_scores(chunk, batch_size, k)
#                     for dids, scores in zip(res_dids, res_scores): 
#                         combined_docs_affscores = list(zip(dids, scores))
#                         combined_sorted = sorted(combined_docs_affscores, key=lambda x:x[1], reverse=True)
#                         dids, scores = zip(*combined_sorted)


#                         fe.write(np.array(dids, dtype=np.uint32).tobytes())
#                         fw.write(np.array(scores, dtype=np.float16).tobytes())


#         # Finally, keep track of metadata about this artefact.
#         with (affinity_path/'pt_meta.json').open('wt') as fout:
#             json.dump({
#                 'type': 'corpus_graph',
#                 'format': 'np_topk',
#                 'doc_count': len(docnos),
#                 'k': k,
#             }, fout)




dsdict = {"docid":list(range(len(docnos)))}

ds  = datasets.Dataset.from_dict(dsdict)
ds = ds.map(lambda example: {"docno": str(example['docid'])}, desc="replacing docids with string docids")
print(ds)

ds = ds.map(affinity_scores, desc="aff scores", batched=True, batch_size=20)

#from_corpus_to_affinity(out_dir_new, edges_path_new, weights_path_new,13, k)


# srun -u --gpus=1 -w gpunode01 python3 save_affinity_graph.py