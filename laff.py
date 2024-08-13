

### Datasets - train msmarco, mine own train/test sets
### Top -k documents from monot5 re-ranking list. 
### fine-tune a bert/tct model with data points (q, d1, d1, S), (q,d3, d4, S)

import warnings
warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.")

from itertools import combinations
from collections import Counter
import numpy as np
import multiprocessing as mp
import pandas as pd
import pickle
import os
import random
from tqdm import tqdm
import torch

import ir_datasets
from pyterrier_adaptive import GAR, CorpusGraph
import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_pisa import PisaIndex, PisaRetrieve

from pyterrier_dr import NumpyIndex, TctColBert, TorchIndex, BiScorer


import datasets
from datasets import load_dataset, Dataset, concatenate_datasets

np.random.seed(42)

def initialize_pyterrier():
    if not pt.started():
        pt.init()



def add_S(example,pipeline, retriever):
    sample_df = pd.DataFrame.from_dict(dict(example))

    ret_result = retriever.transform(sample_df)
    result = pipeline.transform(sample_df)

    ret_df = {qid: group for qid, group in ret_result.groupby('qid')}
    result_df = {qid: group for qid, group in result.groupby('qid')}

    batch_s = []
    batch_score = []
    batch_pos = []
    batch_neg = []

    for qid in example['qid']:
        if qid in ret_df:
            ret_counter = Counter(dict(zip(ret_df[qid].docno, ret_df[qid].score))) 
            pos = [item for item, _ in ret_counter.most_common(5)]
            batch_pos.append(pos)

            all_doc = ret_counter.most_common()
            all_doc.reverse()
            neg = [item for item, _ in all_doc[:5]]
            batch_neg.append(neg)

            res_counter = Counter(dict(zip(result_df[qid].docno, result_df[qid].score))) 
            s = res_counter.most_common(5)
            documents, score = zip(*s)
            batch_s.append(list(documents))
            batch_score.append(list(score))

        else:
            batch_pos.append([])
            batch_neg.append([])
            batch_s.append([])
            batch_score.append([])

    return {"S": batch_s, "score": batch_score, "pseudo_pos": batch_pos, "neg": batch_neg} 


def add_dep_S(example,pipeline, retriever):
    sample_df = pd.DataFrame.from_dict(dict(example))

    ret_result = retriever.transform(sample_df)

    result = pipeline.transform(sample_df)

    ret_df = {qid: group for qid, group in ret_result.groupby('qid')}
    result_df = {qid: group for qid, group in result.groupby('qid')}

    batch_s = []
    batch_score = []
    batch_pos = []
    batch_neg = []

    for qid in example['qid']:
        if qid in ret_df:
            ret_counter = Counter(dict(zip(ret_df[qid].docno, ret_df[qid].score))) 
            s = [item for item, _ in ret_counter.most_common(5)]
            batch_s.append(s)
            res_counter = Counter(dict(zip(result_df[qid].docno, result_df[qid].score))) 
            res_dict = dict(res_counter)
            score = [res_dict[docno] for docno in s]
            batch_score.append(score)

            for docno in s:
                    del res_counter[docno]

            pos = [item for item, _ in res_counter.most_common(5)]
            batch_pos.append(pos)

            all_doc = res_counter.most_common()
            all_doc.reverse()
            neg = [item for item, _ in all_doc[:5]]
            batch_neg.append(neg)

        else:
            batch_pos.append([])
            batch_neg.append([])
            batch_s.append([])
            batch_score.append([])

    #torch.cuda.empty_cache()  # Clear GPU memory cache
    return {"S": batch_s, "score": batch_score, "pseudo_pos": batch_pos, "neg": batch_neg} 

def add_S_ind_data(example,pipeline, retriever):
    sample_df = pd.DataFrame.from_dict(dict(example))

    result = pipeline.transform(sample_df)

    result_df = {qid: group for qid, group in result.groupby('qid')}

    batch_pos = []
    batch_neg = []
    batch_score = []

    for qid in example['qid']:
        if qid in result_df:
            res_counter = Counter(dict(zip(result_df[qid].docno, result_df[qid].score))) 
            p = res_counter.most_common(5)
            n = res_counter.most_common()[:-6:-1]
            pos = [item for item, _ in p]
            neg = [item for item, _ in n]
            score = [item for _, item in p]
            batch_pos.append(pos)
            batch_neg.append(neg)
            batch_score.append(score)

        else:
            batch_pos.append([])
            batch_neg.append([])
            batch_score.append([])

    #torch.cuda.empty_cache()  # Clear GPU memory cache
    return {"pseudo_pos": batch_pos, "neg": batch_neg, "score": batch_score} 





def add_nbh_Q(example,pipeline, scorer,graph):

    sample_df = pd.DataFrame.from_dict(dict(example))

    result = pipeline.transform(sample_df)
    result_df = {qid: group for qid, group in result.groupby('qid')}

    batch_Q = []
    batch_S = []
    batch_score = []
    batch_pos = []
    batch_neg = []
    for qid, query in zip(example['qid'], example['query']):
        if qid in result_df:
            res_counter = Counter(dict(zip(result_df[qid].docno, result_df[qid].score))) 
            s, score = zip(*res_counter.most_common(5))

            neighbors =[]
            for doc in s:
                nbh = graph.neighbours(doc).tolist()
                neighbors.extend(nbh)
                del res_counter[doc]


            ## remove duplicates
            nbh = list(set(neighbors))

            ## remove s from nbh
            nbh = [item for item in nbh if item not in s]

            r_0 = result_df[qid].docno.tolist()

            ## remove r_0 (initial ranked docs) from neighbors
            nbh_not_r_0 = [item for item in nbh if item not in r_0]

            if len(nbh_not_r_0)>0:

                #create batch of docs to re-rank them
                batch = {"docno": nbh_not_r_0}
                batch = pd.DataFrame(batch, columns=['docno'])
                batch['qid'] = qid
                batch['query'] = query

                # score the batch
                rerank_res = scorer(batch)
                rerank_counter = Counter(dict(zip(rerank_res.docno, rerank_res.score)))
                res_counter.update(rerank_counter)

            # choose top 5 docs as positives
            pos = [item for item, _ in res_counter.most_common(5)]

            all_doc = res_counter.most_common()
            all_doc.reverse()

            # choose last 5 docs as negatives
            neg = [item for item, _ in all_doc[:5]]

            ## take rank list as Q
            top_res_counter = res_counter.most_common()
            Q = [item for item, _ in top_res_counter]

            batch_pos.append(pos)
            batch_neg.append(neg)
            batch_Q.append(Q)
            batch_S.append(list(s))
            batch_score.append(list(score))

        else:
            batch_pos.append([])
            batch_neg.append([])
            batch_Q.append([])
            batch_S.append([])
            batch_score.append([])    

    return {"Q":batch_Q,"S":batch_S, "score": batch_score, "pseudo_pos": batch_pos, "neg": batch_neg} 




def preprocess_batch(batch, pipeline, scorer,graph):
    return add_nbh_Q(batch, pipeline, scorer,graph)

# Function to process each shard with batching
def process_shard(shard, gpu_id, dataset_ir):
    device = torch.device(f"cuda:{gpu_id}")
    scorer = pt.text.get_text(dataset_ir, 'text') >> MonoT5ReRanker(verbose=False, batch_size=64,device=device)
    retriever = (
            TctColBert('castorini/tct_colbert-v2-hnp-msmarco',device=device) >> NumpyIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np',verbose=False))

    pipeline = retriever % 100 >> scorer
    graph = CorpusGraph.load("corpusgraph_k128").to_limit_k(20)

    # Apply map with batched=True and batch_size=16
    return shard.map(lambda batch: preprocess_batch(batch, pipeline, scorer, graph), batched=True, batch_size=16)


def process_in_parallel(dataset, num_gpus, dataset_ir):
    # Split dataset into chunks for each GPU
    chunks = [dataset.shard(num_shards=num_gpus, index=i) for i in range(num_gpus)]
    pool = mp.Pool(processes=num_gpus)
    tasks = [(chunk, i,dataset_ir) for i, chunk in enumerate(chunks)]
    results = pool.starmap(process_shard, tasks)
    pool.close()
    pool.join()
    return concatenate_datasets(results)




if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    mp.set_start_method('spawn', force=True)
    warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.")

    initialize_pyterrier()

    dataset_ir = pt.get_dataset('irds:msmarco-passage')
    train_dataset = pt.get_dataset("irds:msmarco-passage/train")

    retriever_name = "tct"

    query_df = train_dataset.get_topics()
    full_query_ds = Dataset.from_pandas(query_df)
    shuffled_ds = full_query_ds.shuffle(seed=42)

    n = 50000 # number of queries to process
    query_ds = shuffled_ds.select(range(n))


    # Process the dataset using multiple GPUs
    if num_gpus > 0:
        processed_dataset = process_in_parallel(query_ds, num_gpus, dataset_ir)
    else:
        processed_dataset = process_shard(query_ds, 0, dataset_ir)

    print("Processing complete")
    print(processed_dataset)
    
    #processed_dataset.save_to_disk(f"data/ms_marcopassage_train_50k_pseudo_pos_neg_nbh_Q_S_ds_{retriever_name}>>monoT5")








