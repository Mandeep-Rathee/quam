import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_dr import NumpyIndex, TctColBert, TorchIndex



dataset = pt.get_dataset('irds:msmarco-passage')


# index_pipeline = (
#     TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
#     TorchIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np', overwrite=True, batch_size=20480))
# index_pipeline.index(dataset.get_corpus_iter())

pipeline_v2_hnp = (
    TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
    NumpyIndex('indices/castorini__tct_colbert-v2-hnp-msmarco.np', verbose=True))



# print(dir(pipeline_v2_hnp))

res = pipeline_v2_hnp.search("clustering hypothesis information retrieval")

print(res)