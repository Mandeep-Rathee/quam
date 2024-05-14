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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy.*")


class AFFM(pt.Transformer):
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
        corpus_graph: "CorpusGraph",
        graph_name: str = "gbm25",
        dl_type: int= 19,
        retriever: str="bm25",
        dataset = None ,
        tokenizer = None,
        edge_mask_learner=None,
        num_results: int = 1000,
        top_int_res: int=50,
        batch_size: Optional[int] = None,
        backfill: bool = True,
        enabled: bool = True,
        use_corpus_graph: bool = True,
        lk: int = 16,
        affm_name: str = None,
        verbose: bool = True):
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
        self.graph_name = graph_name
        self.dl_type = dl_type
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
        self.lk = lk
        self.affm_name = affm_name
        self.verbose = verbose


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Graph-based Adaptive Re-ranking to the provided dataframe. Essentially,
        Algorithm 1 from the paper.
        """
        
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        dict_path = f"scored_docs/ms-marco-passage-{self.graph_name}/dl{self.dl_type}/{self.retriever}>>MonoT5-base-128/pruned_graph_affinity_scores.json"
        pruned_graph_path = f"aff-scored/msmarco-passage-{self.graph_name}/dl{self.dl_type}/{self.retriever}>>MonoT5-base-128"


        #scores_dict = self.load_dict(dict_path)

        try:
            with open(dict_path, 'r') as f:
                scores_dict = json.load(f)
        except FileNotFoundError:
            print(f"The json file {dict_path} is not created. Create an empty json file.")        
        

        try:
            aff_ds = datasets.load_from_disk(pruned_graph_path)
        except FileNotFoundError:
            print("Aff scored path is not available. Please generate and save aff scores using get_scores.py")    

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='affinity based adaptive re-ranking', unit='query')


        for qid in qids:
            query = df[qid]['query'].iloc[0]

            q_aff_ds = aff_ds.filter(lambda example: example["qid"]==qid )

            docid_index = {}
            for i, row in enumerate(q_aff_ds):
                docid_index[row['docno']] = i

            scores = {}
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results {docno: rel score}
            if self.enabled:
                res_map.append(Counter())
            iteration = 0

            #initial_top_docs = res_map[0].most_common(self.top_int_res) ### first choosing top 50 top unraked retrieved docs

            initial_top_docs = {}
            iteration=0    

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

                #print("*"*100)
                #initial_top_docs.update({k: s for k, s  in zip(batch.docno, batch.score)})

                #print(batch)

                # go score the batch of document with the re-ranker
                batch = self.scorer(batch)
                initial_top_docs.update({k: s for k, s  in zip(batch.docno, batch.score)})
                scores.update({k: (s, iteration) for k, s in zip(batch.docno, batch.score)})

                self._drop_docnos_from_counters(batch.docno, res_map)

                if len(scores) < self.num_results and self.enabled: #and iteration%len(res_map)==0:    ##Update frontier only for even iterations. (Search for only 1 hop neighbours.)
                    #self._update_frontier(batch, res_map[1], scores, initial_top_docs,scores_dict)
                    self._update_frontier_corpus_graph(batch, res_map[1],scores, q_aff_ds, docid_index, initial_top_docs,scores_dict)
                iteration+= 1         

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
    

        # Save the updated dictionary to a JSON file
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

    # def generate_result_dict(self, docs, q_aff_ds, docid_index):
    #     result_dict = {}       
    #     for d in docs:
    #         if self.use_corpus_graph:
    #             neighbors, aff_scores = self.corpus_graph.neighbours(d, weights=True)
    #         else:    
    #             selected_row = q_aff_ds[docid_index[d]]
    #             neighbors = selected_row['neighbours']
    #             aff_scores = selected_row['affinity_scores']
    #         result_dict[d] = {"nbh": neighbors, "aff_score": aff_scores}
    #     return result_dict
    
    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]

    """ this function will update the frontier i.e., res_map[1] based on the edges in the Coprus Graph. We will use dense graph, with depth 128.
        Affinity scores for retrived documents are calculated and saved already. For 2 or more hop neighbours we need to calculate them on fly. 
    """
    
    def _update_frontier_corpus_graph(self, scored_batch, frontier, scored_dids, q_aff_ds, docid_index, S, stored_dict):
        """
            Scored_batch: the batch which was scored by scorer in the prevoius Iteration
            frontier: res_map[1] = {"docid" :xpected_aff_score} Either we add the doc to frontier or update the score. 
            q_aff_ds: the part of saved aff scores dataset for query q.
            docid_index: Python dictionary for quick lookUp for doc_id in q_aff_ds
            S: Set $S$ with scores from scorer for documents we have ranked so far.  
        """
        result_dict = {}
        for doc_id in scored_batch.docno:
            if doc_id in docid_index:
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


            result_dict[doc_id] = {"nbh": neighbors, "aff_score": aff_scores}

        """ if we want to use Expected Affinity Model (EAffM), normalize the relevance scores/ node weights."""
        if self.affm_name=="EAffM":
            doc_ids, rel_score = zip(*S.items())
            rel_score = self.softmax(rel_score)
            S = dict(zip(doc_ids, rel_score))

        for doc_id, info in result_dict.items():
            neighbors = info["nbh"]  ##[:self.lk]
            aff_scores = info["aff_score"] ##[:self.lk]

            """ If want to sort the affinity scores"""
            combined_docs_affscores = list(zip(neighbors,aff_scores))
            combined_sorted = sorted(combined_docs_affscores, key=lambda x:x[1], reverse=True)
            top_lk_docs_scores = combined_sorted[:self.lk]
            neighbors, aff_scores = zip(*top_lk_docs_scores)

            """ if we want to use Expected Relevance Model (ERelM)"""
            if self.affm_name=="ERelM":
                aff_scores  = self.softmax(aff_scores) #[x/sum(doc_scores) for x in doc_scores]
            
            for neighbor, aff_score in zip(neighbors, aff_scores):
                if neighbor not in scored_dids:      # Neighboour should not be in scores 
                    if neighbor in frontier:
                        frontier[neighbor]+= aff_score*S[doc_id]   #### A(d,d').R(d)   
                    else:
                        frontier[neighbor] = aff_score*S[doc_id]



    def get_aff_scores(self, initial_top_docs, target_id, scores_dict):

        docs, rel_score = zip(*initial_top_docs.items())

        rel_score = self.softmax(rel_score)    # if want to use expected affinity model

        # if self.use_corpus_graph:
        #         neighbors, aff_scores = self.corpus_graph.neighbours(d, weights=True)
        scores = []
        data_list = []
        for doc in docs:
            if (doc, target_id)  not in scores_dict:
                data_list.append((doc, target_id, 1.0))

        if len(data_list)!=0:
            batch_dataset = MSMARCODataset(data_list, self.dataset,  self.tokenizer, max_length=512)
            loader = DataLoader(batch_dataset, batch_size=1024,shuffle=False)

            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                with torch.no_grad():
                    outputs = self.edge_mask_learner(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                preds = [tensor_item.item() for tensor_item in logits]  
                scores+=preds

        assert len(data_list)==len(scores), f"len of data list={len(data_list)} but the lengh of scores={len(scores)}"        

        for i, score in enumerate(scores):
            scores_dict[data_list[i][:2]]=score

        final_scores = []
        for doc in docs:
            if (doc, target_id) in scores_dict:
                final_scores.append(scores_dict[(doc, target_id)])
            else:
                final_scores.append(scores_dict[(target_id, doc)])

        assert len(rel_score)==len(final_scores), f"len of docs={len(rel_score)} but the lengh of scores={len(final_scores)}"   


        expected_aff_score=0

        #if ERM: #if want to use ecpected relevance model
        #final_scores = self.softmax(final_scores)  

        for x, y in zip(rel_score, final_scores):
            expected_aff_score += x * y

        return expected_aff_score, scores_dict     


    # def _update_fronteir(self,):
    #     docs, rel_score = zip(*unscored_batch)
    #     rel_score = self.softmax(rel_score)
    #     #rel_score = [1.0]*len(rel_score)

    #     doc_score_dict = dict(zip(docs, rel_score))
    #     result_dict = self.generate_result_dict(docs, q_aff_ds, docid_index)


    #     expected_aff_score = defaultdict(float)
    #     for doc, info in result_dict.items():
    #         neighbors = info["nbh"]
    #         doc_scores = info["aff_score"]
    #         for neighbor, score in zip(neighbors, doc_scores):
    #             if neighbor not in docs: 
    #                 expected_aff_score[neighbor]+= score*doc_score_dict[doc]

    #     expected_aff_score_counter = Counter(expected_aff_score)

    #     k_is_c_minus_b=False

    #     if k_is_c_minus_b:
    #         doc_to_select = min(self.num_results-self.batch_size, len(expected_aff_score_counter) )
    #     else:    
    #         doc_to_select = min(self.num_results-self.batch_size, size)

    #     sel_nbh_res = expected_aff_score_counter.most_common(doc_to_select)
    #     final_batch = unscored_batch+sel_nbh_res  


    def _update_frontier(self, scored_batch, frontier, scored_dids, initial_top_docs, scores_dict):
        for doc in scored_batch.docno:
            for target_id in self.corpus_graph.to_limit_k(16).neighbours(doc):
                if target_id not in scored_dids:
                    if target_id not in frontier:
                        frontier[target_id], updated_scores_dict=  self.get_aff_scores(initial_top_docs, target_id, scores_dict)  ### need to find out the expected affinity score based on top_docs_batch
                        scores_dict = updated_scores_dict


    def load_dict(self, filename):
        try:
            with open(filename, 'r') as f:
                # Load JSON data and convert keys from strings to tuples
                return {tuple(map(str, key.split('_'))): value for key, value in json.load(f).items()}
        except FileNotFoundError:
            return {}

    def save_dict(self, filename, data):
        with open(filename, 'w') as f:
            # Convert keys from tuples to strings before saving
            json.dump({('_'.join(map(str, key))): value for key, value in data.items()}, f)



    def get_scores_on_fly(self, doc_id, neighbours):

        scores =[]      
        data_list = []

        for nbh in neighbours:
            data_list.append((doc_id, nbh, 1.0))

        batch_dataset = MSMARCODataset(data_list, self.dataset,  self.tokenizer, max_length=512)
        loader = DataLoader(batch_dataset, batch_size=1024,shuffle=False)

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                outputs = self.edge_mask_learner(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            preds = [tensor_item.item() for tensor_item in logits]  
            scores+=preds

        return scores    
  
        


