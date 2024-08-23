
import warnings
warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.")

from itertools import combinations
from collections import Counter
import numpy as np
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
import torch

import ir_datasets
from pyterrier_adaptive import GAR, CorpusGraph
import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker

from pyterrier_dr import NumpyIndex, TctColBert


import datasets
from datasets import load_dataset, Dataset, concatenate_datasets

np.random.seed(42)

def initialize_pyterrier():
    if not pt.started():
        pt.init()



def add_S(example, pipeline, retriever):
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

        # We found some queries that do not have any relevant documents when BM25 retriever is used. We will add empty lists for them.
        
        else:
            batch_pos.append([])
            batch_neg.append([])
            batch_s.append([])
            batch_score.append([])

    return {"S": batch_s, "score": batch_score, "pseudo_pos": batch_pos, "neg": batch_neg} 



def preprocess_batch(batch, pipeline, scorer):
    return add_S(batch, pipeline, scorer)

# Function to process each shard with batching
def process_shard(shard, gpu_id, dataset_ir):
    device = torch.device(f"cuda:{gpu_id}")
    scorer = pt.text.get_text(dataset_ir, 'text') >> MonoT5ReRanker(verbose=False, batch_size=64,device=device)
    retriever = (
            TctColBert('castorini/tct_colbert-v2-hnp-msmarco',device=device) >> NumpyIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np',verbose=False))

    pipeline = retriever % 100 >> scorer

    # Apply map with batched=True and batch_size=16
    return shard.map(lambda batch: preprocess_batch(batch, pipeline, scorer), batched=True, batch_size=16)


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


    query_df = train_dataset.get_topics()
    full_query_ds = Dataset.from_pandas(query_df)
    shuffled_ds = full_query_ds.shuffle(seed=42)

    """n is the number of queries to process. There are 533k queries in the MSMARCO training dataset. Using all will result into a large train dataset. We choose 50k randomly."""
    
    n = 50000  
    query_ds = shuffled_ds.select(range(n))


    # Process the dataset using multiple GPUs
    if num_gpus > 0:
        processed_dataset = process_in_parallel(query_ds, num_gpus, dataset_ir)
    else:
        processed_dataset = process_shard(query_ds, 0, dataset_ir)

    print("Processing complete")
    print(processed_dataset)
    
    processed_dataset.save_to_disk(f"laff_train_data")








