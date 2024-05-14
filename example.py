import argparse
import os
import numpy as np
import pandas as pd
import sys

import ir_datasets
import torch
import transformers


import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_pisa import PisaIndex
from pyterrier_adaptive import GAR, CorpusGraph, SGAR
from pyterrier_dr import NumpyIndex, TctColBert, TorchIndex
from pyterrier_affinity_model import AFFM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification

from base_models import BinaryClassificationT5Model, BinaryClassificationBertModel


parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=128, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--base_model_name", type=str,default= "bert-base-uncased", help="name of base model")
parser.add_argument("--device", type=str, default="cuda", help="device type/name")
parser.add_argument("--edge_path", type=str, default="edges.u32.np", help="path where edges are stored")
parser.add_argument("--graph_name", type=str, default="gbm25", help="name of the graph")
parser.add_argument("--dl_type", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--seed", type=int, help="seed",default=1234)
parser.add_argument("--sgar", action="store_true", help="if use sparsified GAR")
parser.add_argument("--baseline", action="store_true", help="if use baseline")
parser.add_argument("--affm", action="store_true", help="if use Affinity model")
parser.add_argument("--gar", action="store_true", help="if use GAR model")
parser.add_argument("--batch", type=int, default=16, help="batch size")
parser.add_argument("--budget", type=int, default=100, help="budget c")
parser.add_argument("--top_res", type=int, default=50, help="top intial docs as a seed to the affinity model")
parser.add_argument("--verbose", action="store_true", help="if show progress bar.")
parser.add_argument("--affm_name", type=str,default= "EAffM",help="which Affinity based model to use, ERelM or EAffM ?")
parser.add_argument("--pipeline", type=str, default="bm25>>MonoT5-base", help="name of the pipeline in pyterrier style")
parser.add_argument("--retriever", type=str, default="bm25", help="name of the retriever")



args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
transformers.logging.set_verbosity_error()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


base_model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(base_model_name)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1,torch_dtype=torch.float16)  # Binary classification

model = BinaryClassificationBertModel(base_model)
model.load_state_dict(torch.load(f"models/{base_model_name}_ms_marcopassage_train_random_neg.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()



if args.retriever=="bm25":
    retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
else:
    retriever = (
        TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
        NumpyIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np', verbose=False, cuda=True))

dataset = pt.get_dataset('irds:msmarco-passage')


scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=16)
pipeline = retriever >> scorer


test_dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')


graph_128 = CorpusGraph.load(f"msmarco-passage.{args.graph_name}.1024").to_limit_k(128)

# graph_bm25 = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_bm25_k16').to_limit_k(8)
# graph_tct = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_tcthnp_k16').to_limit_k(8)


  
graph = CorpusGraph.load(f"msmarco-passage.{args.graph_name}.1024").to_limit_k(args.lk)

scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=16)
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  # Display full width of the terminal

from pyterrier.measures import *
dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl_type}/judged')
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  # Display full width of the terminal

exp = pt.Experiment(
    [retriever % args.budget >> scorer,
        retriever  >> GAR(scorer=scorer, corpus_graph=graph,batch_size=args.batch, num_results=args.budget),
        #retriever >>  AFFM(scorer=scorer,corpus_graph=graph_128, num_results=args.budget, batch_size=args.batch, use_corpus_graph=True), 
        retriever >>  AFFM(scorer=scorer,corpus_graph=graph_128,graph_name=args.graph_name , dl_type=args.dl_type,
                        retriever = args.retriever,
                        dataset= docstore, tokenizer= tokenizer,edge_mask_learner=model, 
                        num_results=args.budget,top_int_res=args.top_res, batch_size=args.batch,lk=args.lk, affm_name = args.affm_name,
                        verbose=args.verbose)
        ],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG, nDCG@args.budget,nDCG@10, MAP(rel=2), R(rel=2)@args.budget],
    names=[f"bm25 >> monot5 c={args.budget}", 
            f"bm25>>monot5 GAR+c={args.budget} b={args.batch}",
            #f"bm25>>monot5 AFFM+CG c={args.budget} b={args.batch}",
            f'bm25>>monot5 {args.affm_name} lk={args.lk} c={args.budget} b={args.batch}'
            ]
        #    save_dir = f"results/dl{args.dl_type}/{args.graph_name}/ablation/",
        #    save_mode="overwrite"
)
print(exp.T)
print('*'*100)
