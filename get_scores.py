import torch
import torch.multiprocessing as mp


import ir_datasets
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader 
import json

import datasets

import argparse
import os
import numpy as np
from npids import Lookup
import pandas as pd


import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_pisa import PisaIndex
from pyterrier_adaptive import GAR, CorpusGraph, SGAR
from pyterrier_dr import NumpyIndex, TctColBert, TorchIndex


import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from base_models import  BinaryClassificationBertModel



parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=128, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--base_model_name", type=str,default= "bert-base-uncased", help="name of base model")
parser.add_argument("--device", type=str, default="cuda", help="device type/name")
parser.add_argument("--edge_path", type=str, default="edges.u32.np", help="path where edges are stored")
parser.add_argument("--graph_name", type=str, default="gbm25", help="name of the graph")
parser.add_argument("--seed", type=int, help="seed",default=1234)
parser.add_argument("--dl_type", type=int, default=19, help="dl 19 or 20")
#parser.add_argument("--pipeline", type=str, default="tct>>MonoT5-base", help="name of the pipeline in pyterrier style")
#parser.add_argument("--graph", type=str, default="msmarco-passage.gbm25.1024", help="name of the graph")
parser.add_argument("--retriever", type=str, default="bm25", help="name of the retriever")



args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
transformers.logging.set_verbosity_error()


ds_path = f"run_qrels/msmarco-passage/dl{args.dl_type}/{args.retriever}>>MonoT5-base"


if not os.path.isdir(ds_path):
    dataset = pt.get_dataset('irds:msmarco-passage')
    if args.retriever=="bm25":
        retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
    else:
        retriever = (
        TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
        NumpyIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np', verbose=False, cuda=True))

    scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=16)
    pipeline = retriever >> scorer
    test_data = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl_type}/judged')

    print("run not found.")
    res= pipeline.transform(test_data.get_topics())
    ds  = Dataset.from_pandas(res)
    ds.save_to_disk(ds_path)

else: 
    ds = datasets.load_from_disk(ds_path)


from dataset_utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(ds)

print(ds[0])





### Loading the graph, we start with k =128 and save the similarity scores at onnce. We can re-use them for different k and thresholds


#graph_path = f"msmarco-passage.{args.graph_name}.1024"

graph_path = "corpusgraph_k128"
docnos = Lookup(f"{graph_path}/docnos.npids")
graph_info = json.load(open(f"{graph_path}/pt_meta.json"))

print("docnos loaded")


def edges_data(lk):
    res = np.memmap(f"{graph_path}/{args.edge_path}", mode='r', dtype=np.uint32).reshape(-1, graph_info.get('k')) 
    res = res[:, :lk]
    return res

edges_data_store = edges_data(args.lk)

print("edges data loaded")


def neighbours(docid):
    as_str = isinstance(docid, str)
    if as_str:
      docid = docnos.inv[docid]
    neigh = edges_data_store[docid]
    if as_str:
      neigh = docnos.fwd[neigh]

    return neigh


def add_neighbours(row):
    docid = row["docno"]
    as_str = isinstance(docid, str)
    if as_str:
      docid = docnos.inv[docid]
    neigh = edges_data_store[docid]
    if as_str:
      neigh = docnos.fwd[neigh]


    return {"neighbours": neigh}    


ds  = ds.map(add_neighbours, desc="adding neighborhood pairs in the dataset",num_proc=16) 

print(ds)

print(ds[0])




### Loading model and defining tokenizer
base_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(args.base_model_name)
base_model = BertForSequenceClassification.from_pretrained(args.base_model_name, num_labels=1, torch_dtype=torch.float16)
model = BinaryClassificationBertModel(base_model)
model.load_state_dict(torch.load(f"models/{args.base_model_name}_ms_marcopassage_train_50k_pseudo_pos_neg_S_ds_tct>>monoT5_epoch=5_loss=unbce.pth"))
model.to(device)


dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

#ds = ds.remove_columns(['query', 'text'])
ds = ds.rename_column("score", "rel")

print(ds)

print(ds[0])

def doc_pair_scores(row ):
    sim_scores = []
    data_list = []
    
    for n in row['neighbours']:
        data_list.append((row['docno'], n, 1))

    batch_dataset = MSMARCODataset(data_list, docstore,  tokenizer, max_length=512)
    loader = DataLoader(batch_dataset, batch_size=128,shuffle=False)


    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids=batch['token_type_ids'].to(device) 

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
            logits = outputs.logits

        # Convert logits to probabilities
        #redicted_labels = torch.argmax(logits, dim=1)   
        scores = [tensor_item.item() for tensor_item in logits] 

        sim_scores+=scores

    return {"affinity_scores": sim_scores}




base_path = f"aff-scored/msmarco-passage-{args.graph_name}/dl{args.dl_type}/"
if not os.path.exists(base_path):
   print("path not found")
   os.mkdir(base_path)
   

ds  = ds.map(doc_pair_scores, desc="calculating the sim scores")
print(ds)
print(ds[0])
ds_path = f"aff-scored/msmarco-passage-{args.graph_name}/dl{args.dl_type}/laff_{args.retriever}>>MonoT5-base-{args.lk}"
ds.save_to_disk(ds_path)
