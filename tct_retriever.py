import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_dr import NumpyIndex, TctColBert, TorchIndex, BiScorer
from pyterrier_pisa import PisaIndex, PisaRetrieve
import pandas as pd


dataset = pt.get_dataset('irds:msmarco-passage')


# index_pipeline = (
#     TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
#     TorchIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np', overwrite=True, batch_size=20480))
# index_pipeline.index(dataset.get_corpus_iter())

# pipeline_v2_hnp = (
#     TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
#     NumpyIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np', verbose=True))



# # print(dir(pipeline_v2_hnp))

# res = pipeline_v2_hnp.search("clustering hypothesis information retrieval")

# print(res)


# idx = pt.IterDictIndexer("/home/rathee/quam/indices/bm25_msmarco_passage", verbose=True)

# def generate_documents():
#     for doc in dataset.get_corpus_iter():
#         yield {"docno": doc['docno'], "text": doc['text']}

# idx.index(generate_documents(), fields=['text'], meta=['docno'])



# exit()


# existing_index = pt.IndexFactory.of("/home/rathee/quam/indices/bm25_msmarco_passage")


# ret_scorer = pt.batchretrieve.TextScorer(takes="docs", body_attr="text", wmodel="BM25",background_index= existing_index, properties={'termpipelines' : 'Stopwords,PorterStemmer'})


# print(dir(ret_scorer))

# print(ret_scorer.background_indexref)

# # print(ret_scorer.pr)




df = pd.DataFrame(
    [
        ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
        ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats"],
    ], columns=["qid", "query","docno", "text"])


# rtr = ret_scorer.transform(df)

# print(rtr)
tct_biencoder = BiScorer(bi_encoder_model=TctColBert(verbose=True),batch_size=128)

results = tct_biencoder.transform(df)

print(results)
