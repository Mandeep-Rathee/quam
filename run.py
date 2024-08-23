import argparse
import logging
logging.getLogger("org.terrier.querying.ApplyTermPipeline").setLevel(logging.ERROR)
import numpy as np
import pandas as pd

import ir_datasets
import torch
import transformers


import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_pisa import PisaIndex
from pyterrier_adaptive import  CorpusGraph
from gar_aff import GAR
from pyterrier_quam import QUAM
from transformers import BertTokenizer, BertForSequenceClassification

from base_models import  BinaryClassificationBertModel
#from corpus_graph_q import CorpusGraph

parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=16, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--device", type=str, default="cuda", help="device type/name")
parser.add_argument("--edge_path", type=str, default="edges.u32.np", help="path where edges are stored")
parser.add_argument("--graph_name", type=str, default="gbm25", help="name of the graph")
parser.add_argument("--dl_type", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--seed", type=int, help="seed",default=1234)
parser.add_argument("--batch", type=int, default=16, help="batch size")
parser.add_argument("--budget", type=int, default=100, help="budget c")
parser.add_argument("--top_res", type=int, default=30, help="top intial docs as a seed to the affinity model")
parser.add_argument("--verbose", action="store_true", help="if show progress bar.")
parser.add_argument("--retriever", type=str, default="bm25", help="name of the retriever")
parser.add_argument("--ret_scorer", type=str, default="bm25", help="name of the retriever as a scorer")
parser.add_argument("--use_corpus_graph", action="store_true", help="if use G_c")



args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
transformers.logging.set_verbosity_error()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model_name = "bert-base-uncased" 
tokenizer = BertTokenizer.from_pretrained(base_model_name, torch_dtype=torch.float16)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1,torch_dtype=torch.float16)


"""Define the model and load the pre-trained weights"""

model = BinaryClassificationBertModel(base_model)
model.load_state_dict(torch.load(f"laff_model/bert-base-laff.pth"))
model.to(device)

dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()
retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
dataset = pt.get_dataset('irds:msmarco-passage')



scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=args.batch)
pipeline = retriever >> scorer


"""
We would like to thank the author of the GAR paper for providing the corpus graph. 
We will use the same corpus graph for our experiments. 
We start with the 128 neighbours corpus graph (G_c) and create an affinity Graph (G_a).
"""

graph_128 = CorpusGraph.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128')
graph = CorpusGraph.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128').to_limit_k(16)




pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  # Display full width of the terminal

from pyterrier.measures import *
dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl_type}/judged')
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  # Display full width of the terminal

print(f"retriever={args.retriever} budget c={args.budget} graph={args.graph_name} b={args.batch} lk={args.lk} top={args.top_res}")


exp = pt.Experiment(
    [retriever % args.budget >> scorer,
        retriever >>  GAR(scorer=scorer, corpus_graph=graph_128,graph_name=args.graph_name , dl_type=args.dl_type,
                          num_results=args.budget, laff_scores=False,
                           lk=args.lk, batch_size=args.batch),

        retriever >>  QUAM(scorer=scorer,corpus_graph=graph_128,
                        dataset= docstore, tokenizer= tokenizer,edge_mask_learner=model, 
                        num_results=args.budget,top_int_res=args.top_res, batch_size=args.batch,
                        use_corpus_graph=True, lk=args.lk, verbose=args.verbose),


        retriever  >> GAR(scorer=scorer, corpus_graph=graph_128,graph_name=args.graph_name , dl_type=args.dl_type,
                        retriever_name = args.retriever, 
                         dataset= docstore, tokenizer= tokenizer,edge_mask_learner=model,
                          batch_size=args.batch, num_results=args.budget, laff_scores=True,saved_scores=True,
                          lk=args.lk
                          ),

        retriever >>  QUAM(scorer=scorer,corpus_graph=graph_128,
                        dataset= docstore, tokenizer= tokenizer,edge_mask_learner=model, 
                        num_results=args.budget,top_int_res=args.top_res, batch_size=args.batch,
                        use_corpus_graph=False, lk=args.lk, verbose=args.verbose),

        ],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[f"{args.retriever}_monot5.c{args.budget}", 
            f"GAR.c{args.budget}",            
            f"QuAM.c{args.budget}",
            f"GAR_Laff.c{args.budget}",
            f"QuAM_Laff.c{args.budget}"
            ],
    save_dir = f"saved_pyterrier_runs/{args.graph_name}/dl{args.dl_type}/{args.retriever}/"     
)
print(exp.T)
print('*'*100)
