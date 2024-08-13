from typing import Optional
import numpy as np
import json
from collections import Counter, defaultdict
import pyterrier as pt
import pandas as pd
import ir_datasets
import datasets
logger = ir_datasets.log.easy()
from dataset_utils import *
import torch
import warnings
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy.*")
warnings.filterwarnings("ignore",message="WARN org.terrier.querying.ApplyTermPipeline - The index has no termpipelines configuration, and no control configuration is found. Defaulting to global termpipelines configuration of 'Stopwords,PorterStemmer'. Set a termpipelines control to remove this warning.")

use_ret_scores = False
use_rel_scores = True

class QUAM(pt.Transformer):
    """
    Required input columns: ['qid', 'query', 'docno', 'score', 'rank']
    Output columns: ['qid', 'query', 'docno', 'score', 'rank', 'iteration']
    where iteration defines the batch number which identified the document. Specifically
    even=initial retrieval   odd=corpus graph    -1=backfilled
    
    """
    def __init__(self,
        scorer: pt.Transformer,
        corpus_graph: "CorpusGraph",
        graph_name: str = "gbm25",
        dl_type: int= 19,
        retriever_name: str="bm25",
        retriever = None,
        dataset = None ,
        tokenizer = None,
        edge_mask_learner=None,
        num_results: int = 1000,
        top_int_res: int=300,
        batch_size: Optional[int] = None,
        backfill: bool = True,
        enabled: bool = True,
        use_corpus_graph: bool = False,
        use_int:bool = False,
        lk: int = 16,
        saved_scores:bool = False,
        alpha: float=0.5,
        affm_name: str = None,
        verbose: bool = True):
        """
            Quam init method
            Args:
                scorer(pyterrier.Transformer): A transformer that scores query-document pairs. It will only be provided with ['qid, 'query', 'docno', 'score'].
                corpus_graph(pyterrier_adaptive.CorpusGraph): A graph of the corpus, enabling quick lookups of nearest neighbours
                num_results(int): The maximum number of documents to score (called "budget" and $c$ in the paper)
                batch_size(int): The number of documents to score at once (called $b$ in the paper). If not provided, will attempt to use the batch size from the scorer
                backfill(bool): If True, always include all documents from the initial stage, even if they were not re-scored
                enabled(bool): If False, perform re-ranking without using the corpus graph
                verbose(bool): If True, print progress information
        """
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.graph_name = graph_name
        self.dl_type = dl_type
        self.retriever_name = retriever_name
        self.retriever = retriever
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.edge_mask_learner = edge_mask_learner
        self.num_results = num_results
        self.top_int_res = top_int_res
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.enabled = enabled
        self.use_corpus_graph = use_corpus_graph
        self.use_int = use_int
        self.lk = lk
        self.saved_scores = saved_scores
        self.alpha = alpha
        self.affm_name = affm_name
        self.verbose = verbose


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Query Affinity Modeling Quam applies the Algorithm 1 from the paper.
        """
        
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        dict_path = f"scored_docs/ms-marco-passage-{self.graph_name}/dl{self.dl_type}/{self.retriever_name}>>MonoT5-base-128/laff_scores.json" 
        pruned_graph_path = f"aff-scored/msmarco-passage-{self.graph_name}/dl{self.dl_type}/laff_{self.retriever_name}>>MonoT5-base-128"


        scores_dict = {}
        if self.saved_scores:
            with open(dict_path, 'r') as f:
                scores_dict = json.load(f)
            aff_ds = datasets.load_from_disk(pruned_graph_path)


        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='affinity based adaptive re-ranking', unit='query')

        for qid in qids:

            if not self.saved_scores:
                docid_index = {}
                q_aff_ds = None
            else:
                docid_index = {}
                q_aff_ds = aff_ds.filter(lambda example: example["qid"]==qid )
                for i, row in enumerate(q_aff_ds):
                    docid_index[row['docno']] = i

            
            scores = {}
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results {docno: rel score}
            if self.enabled:
                res_map.append(Counter())
            iteration = 0

            r1_upto_now = {}
            iteration=0  
            query = df[qid]['query'].iloc[0]

            while len(scores) < self.num_results and any(r for r in res_map):
                if len(res_map[iteration%len(res_map)])==0:
                    iteration+=1
                    continue

                this_res = res_map[iteration%len(res_map)] # alternate between the initial ranking and frontier
                size = min(self.batch_size, self.num_results - len(scores)) # get either the batch size or remaining budget (whichever is smaller)

                # build batch of documents to score in this round
                
                batch = this_res.most_common(size)

                batch = pd.DataFrame(batch, columns=['docno', 'score'])
                batch['qid'] = qid
                batch['query'] = query
                    

                # go score the batch of document with the re-ranker
                batch = self.scorer(batch)

                scores.update({k: (s, iteration) for k, s in zip(batch.docno, batch.score)})

                self._drop_docnos_from_counters(batch.docno, res_map)

                if len(scores) < self.num_results and self.enabled and new_docs is not None: 

                    r1_upto_now.update({k: s for k, s  in zip(batch.docno, batch.score)})    # Re-ranked doccumnets (R1) so far 
                    S = dict(Counter(r1_upto_now).most_common(self.top_k_docs))       # Take top_int_res(hyper-parameter) documents from R1
                    recent_docs = set(batch.docno)
                    new_docs = recent_docs.intersection(S)  ### Find newly re-ranked documents in S    
                    
                    if new_docs is not None:
                        self._update_frontier_corpus_graph(new_docs, res_map[1],scores, q_aff_ds, docid_index, S, scores_dict)

                iteration+=1   

            
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, (score, i) in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
                result['iteration'].append(i)   

            # Backfill unscored items
            if self.backfill and len(scores) < self.num_results:
                last_score = result['score'][-1] if result['score'] else 0.
                count = min(self.num_results - len(scores), len(res_map[0]))
                result['qid'].append(np.full(count, qid))
                result['query'].append(np.full(count, query))
                result['rank'].append(np.arange(len(scores), len(scores) + count))
                for i, (did, score) in enumerate(res_map[0].most_common()):
                    if i >= count:
                        break
                    result['docno'].append(did)
                    result['score'].append(last_score - 1 - i)
                    result['iteration'].append(-1)
    


        """ Save the updated dictionary to a JSON file """

        if self.saved_scores:
            with open(dict_path, 'w') as f:
                json.dump(scores_dict, f)

        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })    
    
    def softmax(self, logits):
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    
    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]

    """ this function will update the frontier i.e., res_map[1] based on the edges in the Coprus Graph G_c or Affinity Graph G_a. We will use dense graph, with depth 128.
        Affinity scores for retrived documents are calculated and saved already. For 2 or more hop neighbours we need to calculate them on fly. 
    """
    
    def _update_frontier_corpus_graph(self, scored_batch, frontier, scored_dids, q_aff_ds, docid_index, S, stored_dict):
        """
            Scored_batch: the documents from prevoius Iteration's batch which are in top_res (topk documents from R1)
            frontier: res_map[1] = {"docid" :xpected_aff_score} Either we add the doc to frontier or update the score. 
            q_aff_ds: the part of saved aff scores dataset for query q.
            docid_index: Python dictionary for quick lookUp for doc_id in q_aff_ds
            S: Set $S$ with scores from scorer for documents in top_res.  
        """

        """ if we want to use Set Affinity, normalize the relevance scores/ node weights."""

        doc_ids, rel_score = zip(*S.items())
        rel_score = self.softmax(np.array(rel_score))
        S = dict(zip(doc_ids, rel_score))


        for doc_id in scored_batch:
            if self.use_corpus_graph:     # Use G_c (Corpus Graph) for edge weights (affinity scores)
                neighbors, aff_scores = self.corpus_graph.to_limit_k(self.lk).neighbours(doc_id, True)
                if self.graph_name=="gtcthnp":
                     aff_scores =  [(x - min(aff_scores)) / (max(aff_scores) - min(aff_scores)) for x in aff_scores]
                neighbors = neighbors.tolist()

            else:
                """First we try to look in afinity ds or stored dict. 
                Otherwise we generate affinity scores on fly and update the stored dict.
                """
                if doc_id in docid_index:   # Use G_a (Affinity Graph) for edge weights (affinity scores)
                    selected_row = q_aff_ds[docid_index[doc_id]]
                    neighbors = selected_row['neighbours']
                    aff_scores = selected_row['affinity_scores']
                elif doc_id in stored_dict:
                    neighbors = stored_dict[doc_id]["neighbours"]
                    aff_scores = stored_dict[doc_id]["affinity_scores"]
                else:
                    neighbors = self.corpus_graph.neighbours(doc_id).tolist()
                    aff_scores = self.get_scores_on_fly(doc_id, neighbors)  ### Should come from either saved or compute the scores. 
                    stored_dict[doc_id] = {"neighbours": neighbors, "affinity_scores":aff_scores}  


            """ If affinity scores are not sorted, sort them and take top k neighbours. """  

            combined_docs_affscores = list(zip(neighbors,aff_scores))
            combined_sorted = sorted(combined_docs_affscores, key=lambda x:x[1], reverse=True)
            top_lk_docs_scores = combined_sorted[:self.lk]
            neighbors, aff_scores = zip(*top_lk_docs_scores)


            """for each neighbour, calculate or update the set affinity score and update the frontier."""

            for neighbor, aff_score in zip(neighbors, aff_scores):
                s_doc = S[doc_id]
                if neighbor not in scored_dids:      # Neighboour should not be in scores 
                    frontier[neighbor]+=aff_score*s_doc   #### f(d,d').R(d)


    def get_scores_on_fly(self, doc_id, neighbours):

        scores =[]      
        data_list = []

        for nbh in neighbours:
            data_list.append((doc_id, nbh, 1.0))

        batch_dataset = MSMARCODataset(data_list, self.dataset,  self.tokenizer)
        loader = DataLoader(batch_dataset, batch_size=1024,shuffle=False)

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids=batch['token_type_ids'].to(device) 

            with torch.no_grad():
                outputs = self.edge_mask_learner(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
                logits = outputs.logits

            preds = [tensor_item.item() for tensor_item in logits]  
            scores+=preds

        return scores    
  



