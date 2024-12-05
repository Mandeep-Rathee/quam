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
import pyterrier_alpha as pta
from gar_aff import GAR
from pyterrier_quam import QUAM


parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=16, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--graph_name", type=str, default="gbm25", help="name of the graph")
parser.add_argument("--dl_type", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--seed", type=int, help="seed",default=1234)
parser.add_argument("--batch", type=int, default=16, help="batch size")
parser.add_argument("--budget", type=int, default=100, help="budget c")
parser.add_argument("--s", type=int, default=30, help="top s docs (S) to calculate the set affinity.")
parser.add_argument("--verbose", action="store_true", help="if show progress bar.")
parser.add_argument("--retriever", type=str, default="bm25", help="name of the retriever")
parser.add_argument("--ret_scorer", type=str, default="bm25", help="name of the retriever as a scorer")



args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
transformers.logging.set_verbosity_error()

# Load the dataset and the retriever
retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
dataset = pt.get_dataset('irds:msmarco-passage')


scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=args.batch)
pipeline = retriever >> scorer


# Load the graphs from the hub

corpus_graph = CorpusGraph.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128').to_limit_k(args.lk)
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff').to_limit_k(args.lk)



from pyterrier.measures import *
dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl_type}/judged')
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  # Display full width of the terminal

print(f"retriever={args.retriever} budget c={args.budget} graph={args.graph_name} b={args.batch} lk={args.lk} top={args.s}")


exp = pt.Experiment(
    [retriever % args.budget >> scorer,
        retriever >>  GAR(scorer=scorer, corpus_graph = corpus_graph, num_results= args.budget, 
                        batch_size=args.batch),

        retriever >>  QUAM(scorer=scorer,corpus_graph=corpus_graph,
                        num_results=args.budget, top_k_docs=args.s, batch_size=args.batch,
                       verbose=args.verbose),

        retriever  >> GAR(scorer=scorer, corpus_graph=laff_graph, 
                          batch_size=args.batch, num_results=args.budget                          ),

        retriever >>  QUAM(scorer=scorer,corpus_graph=laff_graph,
                        num_results=args.budget, top_k_docs=args.s, batch_size=args.batch,
                         verbose=args.verbose),

        ],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[f"{args.retriever}_monot5.c{args.budget}", 
            f"GAR.c{args.budget}",    # GAR_bm25          
            f"QuAM.c{args.budget}",  # GAR_bm25 + SetAff
            f"GAR_Laff.c{args.budget}", # GAR_bm25 + Laff
            f"QuAM_Laff.c{args.budget}" # Quam_bm25
            ],
    #save_dir = f"saved_pyterrier_runs/{args.graph_name}/dl{args.dl_type}/{args.retriever}/"     # If you do not want to use the saved runs, please comment this line.
)
print(exp.T)
print('*'*100)

