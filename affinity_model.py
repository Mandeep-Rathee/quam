import ir_datasets
from datasets import Dataset, load_dataset
import datasets
from typing import Union, Tuple, List


import argparse
from collections import Counter, defaultdict
import os
import numpy as np
from npids import Lookup
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch_geometric.utils as ut

import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_pisa import PisaIndex
from pyterrier_adaptive import GAR, CorpusGraph, SGAR

import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from base_models import  BinaryClassificationBertModel
from dataset_utils import *
from graph_plots_utils import plot_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=128, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--base_model_name", type=str,default= "bert-base-uncased", help="name of base model")
parser.add_argument("--device", type=str, default="cuda", help="device type/name")
parser.add_argument("--edge_path", type=str, default="edges.u32.np", help="path where edges are stored")
parser.add_argument("--graph", type=str, default="msmarco-passage.gbm25.1024", help="name of the graph")
parser.add_argument("--seed", type=int, help="seed",default=1234)


args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
transformers.logging.set_verbosity_error()



docnos = Lookup(f"{args.graph}/docnos.npids")



### Loading model and defining tokenizer
base_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(args.base_model_name)
base_model = BertForSequenceClassification.from_pretrained(args.base_model_name, num_labels=1)
model = BinaryClassificationBertModel(base_model)
model.load_state_dict(torch.load(f"models/{args.base_model_name}_ms_marcopassage_train_random_neg.pth"))
model.to(device)


ds_path = f"aff-scored/msmarco-passage-gbm25/bm25>>MonoT5-base-{args.lk}"

aff_ds = datasets.load_from_disk(ds_path)

graph = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_bm25_k16').to_limit_k(4)


num_results= 8
batch_size= 4

dataset = pt.get_dataset('irds:msmarco-passage')
retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=16)

test_data = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')


"""even=initial retrieval   odd=corpus graph    -1=backfilled"""


ds = datasets.load_from_disk("run_qrels/msmarco-passage/bm25>>MonoT5-base")




#aff_ds  = aff_ds.map(add_query_text, desc="adding query text")
# aff_ds = aff_ds.map(reduce_nbh, desc="reducing the neighbourhood")

df = ds.to_pandas()



aff_ds = aff_ds.remove_columns(['query', 'text', 'rank'])


def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def generate_result_dict(docs, q_aff_ds, docid_index):
    result_dict = {}       
    for d in docs:
        selected_row = q_aff_ds[docid_index[d]]
        neighbors = selected_row['neighbours'][:4]
        aff_scores = selected_row['affinity_scores'][:4]
        result_dict[d] = {"nbh": neighbors, "aff_score": aff_scores}
    return result_dict


def aff_transfom(df: pd.DataFrame) -> pd.DataFrame:

    result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

    df = dict(iter(df.groupby(by=['qid'])))
    qids = df.keys()
    
    for qid in list(qids)[:1]:
        query = df[qid]['query'].iloc[0]

        qdf = df[qid]
        q_aff_ds = aff_ds.filter(lambda example: example["qid"]==qid)
        print(len(q_aff_ds))

        docid_index = {}
        for i, row in enumerate(q_aff_ds):
            docid_index[row['docno']] = i

        scores = {}
        res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results {docno: rel score}
        iteration = 0
        this_res = res_map[0] 
        size = min(batch_size, num_results - len(scores)) # get either the batch size or remaining budget (whichever is smaller)

        # build batch of documents to score in this round
        unscored_batch = this_res.most_common(size)

        print(unscored_batch)
        print("*"*100)
        docs, rel_score = zip(*unscored_batch)
        #rel_score = softmax(rel_score)
        doc_score_dict = dict(zip(docs, rel_score))

        result_dict = {}       
        for d in docs:
            selected_row = q_aff_ds[docid_index[d]]
            neighbors = selected_row['neighbours'][:4]
            aff_scores = selected_row['affinity_scores'][:4]
            result_dict[d] = {"nbh": neighbors, "aff_score": aff_scores}


        plot_graph(result_dict, doc_score_dict, f"plots/graphs/qid={qid}_iter=0.pdf", 0)


        expected_aff_score = defaultdict(float)
        for doc, info in result_dict.items():
            neighbors = info["nbh"]
            doc_scores = info["aff_score"]
            norm_doc_scores  = [x/sum(doc_scores) for x in doc_scores]
            
            for neighbor, score in zip(neighbors, norm_doc_scores):
                if neighbor not in docs: 
                    expected_aff_score[neighbor]+= score*doc_score_dict[doc]


        print("*"*100)

        print(dict(expected_aff_score))

        doc = dict(expected_aff_score).get('8760870')
        print(doc)

        node_weights = {**doc_score_dict, **dict(expected_aff_score)}
        print(node_weights)


        plot_graph(result_dict, node_weights, f"plots/graphs/qid={qid}_iter=1.pdf", 1)


        exit()


 
        batch = pd.DataFrame(unscored_batch, columns=['docno', 'score'])
        batch['qid'] = qid
        batch['query'] = query

        # go score the batch of document with the re-ranker
        batch = scorer(batch)
        scores.update({k: (s, iteration) for k, s in zip(batch.docno, batch.score)})

        result['qid'].append(np.full(len(scores), qid))
        result['query'].append(np.full(len(scores), query))
        result['rank'].append(np.arange(len(scores)))
        for did, (score, i) in Counter(scores).most_common():
            result['docno'].append(did)
            result['score'].append(score)
            result['iteration'].append(i)   

    
    return result

result = aff_transfom(df)
print(result)




        # edge_data = []
        # src = []
        # index = []
        # edge_attr = []

        # for doc in docs:
        #     selected_row = q_aff_ds[docid_index[doc]]
        #     neighbors = selected_row['neighbours'][:4]
        #     aff_scores = selected_row['affinity_scores'][:4]
        #     for nbh, aff in zip(neighbors, aff_scores):
        #         src.append(int(doc))
        #         index.append(int(nbh))
        #         edge_attr.append(aff)




        # edge_index = torch.tensor([src,index], dtype=torch.long)
        # print(edge_index)

        # print("is undirected:",ut.is_undirected(edge_index))
        # print("contain self loops:",ut.contains_self_loops(edge_index))
        # print()
        # print(edge_index.shape)

        # print(edge_attr)

        # new_edge_index = ut.to_undirected(edge_index)
        # print(new_edge_index)
        # print(new_edge_index.shape)

