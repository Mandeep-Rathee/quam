from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm

import ir_datasets
from pyterrier_adaptive import GAR, CorpusGraph
import pyterrier as pt
pt.init()



dataset = ir_datasets.load("msmarco-passage/train")
dataset_2 = pt.get_dataset('irds:msmarco-passage/train')
graph = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_tcthnp_k16').to_limit_k(8)
# print(dataset.docs)
# print(dataset_2)

doc_store = dataset.docs_store()

doc_dict = doc_store.get_many(['0', '100'])
text_list = [doc.text for doc in doc_dict.values()]   

query_id_to_text = {query.query_id: query.text for query in dataset.queries_iter()}
doc_ids = [doc.doc_id for doc in dataset.docs_iter()]

print(len(doc_ids)) ## 88,41,823

# queries_df = pd.DataFrame(dataset.queries_iter())
# query_id_to_text = dict(zip(queries_df['query_id'], queries_df['text']))


rel_doc_count = {}
for qrel in dataset.qrels_iter():
    query_id = qrel.query_id
    if query_id in rel_doc_count:
        if qrel.doc_id not in rel_doc_count[query_id]:
            rel_doc_count[query_id].append(qrel.doc_id)
    else:
        rel_doc_count[query_id] = [qrel.doc_id]


training_data_list = []

for query_id, doc_list in tqdm(rel_doc_count.items()):
    counter = Counter(training_data_list)
    if len(doc_list)>1:
        pairs = list(combinations(doc_list, 2))
        for pair in pairs:
            training_data_list.append((pair[0], pair[1], 1))

            # for each positive, choose a random negative
            random_neg = random.choice(doc_ids)
            if (pair[0], random_neg, 0) and (pair[0], random_neg, 1) not in counter:
                training_data_list.append((pair[0], random_neg, 0))



exit()
with open('data/ms_marcopassage_train_tct_pos_and_neg.pkl', 'wb') as f:
    pickle.dump(training_data_list, f)





#positive_samples = tuple(zip(query_ids, passage_s_id, passage_t_id))


### Generate Negative samples

    ### nbrnp --> neighbors & in retrieval & not positive
    # elif len(doc_list)==1:
    #     ## adding positives and negatives based on neighbours in graph
    #     ps_id = doc_list[0]
    #     neighbours, _ = graph.neighbours(docid = ps_id, weights=True)   
    #     k=2
    #     for nbh in neighbours[:k]:
    #         if (ps_id,nbh,1) not in training_data_list:
    #             training_data_list.append((ps_id,nbh,1))
    #     for nc in neighbours[-k:]:            
    #         training_data_list.append((ps_id, nc, 0))

# for query_id, doc_list in tqdm(rel_doc_count.items()):
#     if len(doc_list)==1:
#         q_text = query_id_to_text.get(query_id)
#         ps_id = doc_list[0]
#         neighbours, weight = graph.neighbours(docid = ps_id, weights=True)   
#         nbrnp = neighbours[-k:]
#         ### nbrnp --> neighbors & in retrieval & not positive
#         for nc in nbrnp:
#             if (query_id, ps_id, nc) not in positive_samples: 


# for query_id, doc_list in tqdm(rel_doc_count.items()):

#     if len(doc_list)==1:
#         q_text = query_id_to_text.get(query_id)
#         ps_id = doc_list[0]
#         result = retriever.search(q_text)
#         result_doc = result.iloc[:,2]
#         neighbours, weight = graph.neighbours(docid = ps_id, weights=True)   
#         nbrnp = list(set(neighbours[-k:]) & set(result_doc))
#         ### nbrnp --> neighbors & in retrieval & not positive
#         for nc in nbrnp[:5]:
#             if (query_id, ps_id, nc) not in positive_samples: 

#         #### rnp --> retrieval & not positive
#         temp_neg_tuple  = tuple(temp_neg_list)
#         rnp = list(set(result_doc)- set(neighbours[-k:]))
        
#         for nc in rnp[:5]:
#             if (query_id, ps_id, nc) not in positive_samples and (query_id, ps_id, nc) not in temp_neg_tuple: 

        ### nbnp --> neighbors & not positive    