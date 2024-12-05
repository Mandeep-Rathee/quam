from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
logger = ir_datasets.log.easy()
import torch
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy.*")
warnings.filterwarnings("ignore",message="WARN org.terrier.querying.ApplyTermPipeline - The index has no termpipelines configuration, and no control configuration is found. Defaulting to global termpipelines configuration of 'Stopwords,PorterStemmer'. Set a termpipelines control to remove this warning.")


class QUAM(pt.Transformer):
    """This class is based on pyterrier_adaptive (graph based adaptive re-ranking) https://github.com/terrierteam/pyterrier_adaptive
    Required input columns: ['qid', 'query', 'docno', 'score', 'rank']
    Output columns: ['qid', 'query', 'docno', 'score', 'rank', 'iteration']
    where iteration defines the batch number which identified the document. Specifically
    even=initial retrieval   odd=corpus graph    -1=backfilled
    
    """
    def __init__(self,
        scorer: pt.Transformer,
        corpus_graph: "CorpusGraph",
        num_results: int = 1000,
        top_k_docs: int=300,
        batch_size: Optional[int] = None,
        backfill: bool = True,
        enabled: bool = True,
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
        self.num_results = num_results
        self.top_k_docs = top_k_docs
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.enabled = enabled
        self.verbose = verbose


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Query Affinity Modeling Quam applies the Algorithm 1 from the paper.
        """
        
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='affinity based adaptive re-ranking', unit='query')

        for qid in qids:

            scores = {}
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results {docno: rel score}
            if self.enabled:
                res_map.append(Counter())

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

                if len(scores) < self.num_results and self.enabled: 

                    r1_upto_now.update({k: s for k, s  in zip(batch.docno, batch.score)})    # Re-ranked doccumnets (R1) so far 
                    S = dict(Counter(r1_upto_now).most_common(self.top_k_docs))       # Take top s(hyper-parameter) documents from R1
                    recent_docs = set(batch.docno)
                    new_docs = recent_docs.intersection(S)  ### Find newly re-ranked documents in S    
                    
                    if new_docs is not None:
                        self._update_frontier_corpus_graph(new_docs, res_map[1],scores, S)

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

    """ 
    this function will update the frontier i.e., res_map[1] based on the edges in the Coprus Graph G_c or Affinity Graph G_a.
    
    """
    
    def _update_frontier_corpus_graph(self, scored_batch, frontier, scored_dids, S):
        """
            Scored_batch: the documents from prevoius Iteration's batch which are in top_res (topk documents from R1)
            frontier: res_map[1] = {"docid" :set_aff} Either we add the doc to frontier or update the score. 
            S: Set $S$ with scores from scorer for top k documents from R1.  
        """

        """ if we want to use Set Affinity, normalize the relevance scores/ node weights."""

        doc_ids, rel_score = zip(*S.items())
        rel_score = self.softmax(np.array(rel_score))
        S = dict(zip(doc_ids, rel_score))


        for doc_id in scored_batch:

            neighbors, aff_scores = self.corpus_graph.neighbours(doc_id, True)

            """for each neighbour, calculate or update the set affinity score and update the frontier."""

            for neighbor, aff_score in zip(neighbors, aff_scores):
                s_doc = S[doc_id]
                if neighbor not in scored_dids:      # Neighboour should not be in scores 
                    frontier[neighbor]+=aff_score*s_doc   #### f(d,d').R(d)


