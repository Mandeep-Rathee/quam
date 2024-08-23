from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import datasets
import time
import gc



logger = ir_datasets.log.easy()

import json

import torch
from dataset_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAR(pt.Transformer):
    """
    A transformer that implements the Graph-based Adaptive Re-ranker algorithm from
    MacAvaney et al. "Adaptive Re-Ranking with a Corpus Graph" CIKM 2022.

    Required input columns: ['qid', 'query', 'docno', 'score', 'rank']
    Output columns: ['qid', 'query', 'docno', 'score', 'rank', 'iteration']
    where iteration defines the batch number which identified the document. Specifically
    even=initial retrieval   odd=corpus graph    -1=backfilled
    
    """
    def __init__(self,
        scorer: pt.Transformer,
        corpus_graph: 'CorpusGraph',
        num_results: int = 1000,
        batch_size: Optional[int] = None,
        retriever = None,
        dataset = None ,
        tokenizer = None,
        laff_model=None,
        use_int:bool = False,
        laff_scores:bool = False,
        saved_scores:bool = False,
        lk: int = 16,
        backfill: bool = True,
        enabled: bool = True,
        verbose: bool = False):
        """
            GAR init method
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
        self.num_results = num_results
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.retriever = retriever
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.laff_model = laff_model
        self.laff_scores = laff_scores
        self.lk = lk
        self.saved_scores = saved_scores
        self.backfill = backfill
        self.enabled = enabled
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Graph-based Adaptive Re-ranking to the provided dataframe. Essentially,
        Algorithm 1 from the paper.
        """
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='adaptive re-ranking', unit='query')

        print("Starting GAR")
        for qid in qids:

            query = df[qid]['query'].iloc[0]

            scores = {}
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results
            if self.enabled:
                res_map.append(Counter()) # frontier
            frontier_data = {'minscore': float('inf')}
            iteration = 0
            while len(scores) < self.num_results and any(r for r in res_map):
                if len(res_map[iteration%len(res_map)]) == 0:
                    # if there's nothing available for the one we select, skip this iteration (i.e., move on to the next one)
                    iteration += 1
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
                if len(scores) < self.num_results and self.enabled:
                        self._update_frontier(batch, res_map[1], frontier_data, scores)
                iteration += 1


            # Add scored items to results
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



        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })

    def _update_frontier(self, scored_batch, frontier, frontier_data, scored_dids):
        remaining_budget = self.num_results - len(scored_dids)
        for score, did in sorted(zip(scored_batch.score, scored_batch.docno), reverse=True):
            if len(frontier) < remaining_budget or score >= frontier_data['minscore']:
                hit = False
                if self.laff_scores:
                    target_neighbors = self.laff_based_neighbors(did)    
                else:
                    target_neighbors = self.corpus_graph.to_limit_k(self.lk).neighbours(did)
                for target_did in target_neighbors:
                    if target_did not in scored_dids:
                        if target_did not in frontier or score > frontier[target_did]:   ## Either you update the frontier target id score or add target id to the frontier
                            frontier[target_did] = score
                            hit = True
                if hit and score < frontier_data['minscore']:
                    frontier_data['minscore'] = score

    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]

    
    def laff_based_neighbors(self, doc_id):
        
        neighbors = self.corpus_graph.neighbours(doc_id).tolist()
        aff_scores = self.get_scores_on_fly(doc_id, neighbors)  ### Should come from either saved or compute the scores. 

        """ If want to sort the affinity scores"""
        combined_docs_affscores = list(zip(neighbors,aff_scores))
        combined_sorted = sorted(combined_docs_affscores, key=lambda x:x[1], reverse=True)
        top_lk_docs_scores = combined_sorted[:self.lk]
        neighbors, aff_scores = zip(*top_lk_docs_scores)

        return neighbors
    
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
                outputs =  self.laff_model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
                logits = outputs.logits

            preds = [tensor_item.item() for tensor_item in logits]  
            scores+=preds

        return scores    
  