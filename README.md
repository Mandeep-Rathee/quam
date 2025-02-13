# QuAM: Adaptive Retrieval through Query Affinity Modelling


This is Github repository for our paper [Quam: Adaptive Retrieval through Query Affinity Modelling](https://arxiv.org/pdf/2410.20286)  accepted in 18th ACM International Conference on Web Search and Data Mining (WSDM 2025), Hannover, Germany, 10-14 Mar 2025. 

<p align="center">
  <img src="quam_main_fig.jpg" />
</p>

## Setup

### Requirements
We have added all dependencies in requirements.txt file which can be downloaded as follows:

```
pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git
pip install --upgrade git+https://github.com/terrierteam/pyterrier_adaptive.git
pip install pyterrier_pisa==0.0.6
```

### :file_folder: Files Structure

```
├── data/
│    ├── laff_train_data/
├── laff_model/
├── saved_pyterrier_runs/
│    ├── gbm25/
│    ├── gtcthnp/
├── base_models.py
├── dataset_utils.py
├── gar_aff.py
├── laff.py
├── pyterrier_quam.py
├── quam_main_fig.jpg
├── requirements.txt
├── run.py
├── train_laff.py
└── README.md
```

## Corpus and Affinity Graph
We use the same corpus graph from the [GAR](https://arxiv.org/pdf/2208.08942) paper and we release our laff scores based affinity graph. 
For instance, the bm25 based corpus and affinity graph can be downloaded using:
```
import pyterrier_alpha as pta
corpus_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128')
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff')

```


## Training Data for Learnt Affinity Model
If you want to generate own training dataset and train the affinity model, the training data can be created using the `laff.py` file. Alternatively, we release the training dataset for learnt affinity model from [Huggingface](https://huggingface.co/mandeep-rathee/laff-model/tree/main/data/laff_train_data). The dataset has following files:

1. data-00000-of-00001.arrow
2. dataset_info.json
3. state.json

Please download all 3 files in the `data/laff_train_data/` folder. Further, the dataset can be loaded as

```
import datasets
ds = datasets.load_from_disk("data/laff_train_data")
```

## Learnt Affinity Model
If you want to use the learnt affinity model for document-document similarity, you can train using the `train_laff.py` file. Alternatively, we have released the model's weights and can be downloaded from [huggingface](https://huggingface.co/mandeep-rathee/laff-model/tree/main). The corresponding file is:

- bert-base-laff.pth


Alternatively, use the following code:

```
from huggingface_hub import hf_hub_download
file_path = hf_hub_download(repo_id="mandeep-rathee/laff-model", filename="bert-base-laff.pth")
```

The model can loaded as follow:

```
from transformers import BertTokenizer, BertForSequenceClassification
from base_models import  BinaryClassificationBertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model_name = "bert-base-uncased" 
tokenizer = BertTokenizer.from_pretrained(base_model_name, torch_dtype=torch.float16)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1,torch_dtype=torch.float16)

model = BinaryClassificationBertModel(base_model)
model.load_state_dict(torch.load(file_path))  ## or saved model path
model.to(device)
```

## Reproduction

Our results can be reproduced by using the `run.py` file. Additionally, we have also provided the saved runs in the  `saved_pyterrier_runs/` folder.

We use the following combinations of budget c and |S|=s:

|budget | s |
| ------ | --- |
| 50 | 10 |
| 100 | 30|
| 1000 | 300|


To reproduce the results for bm25 retriever and corpus graph for TREC DL'19 , run

```
python3 run.py --budget 50 --s 10 --verbose --dl_type 19
```



## Citation
```
@article{rathee2024quam,
  title={Quam: Adaptive Retrieval through Query Affinity Modelling},
  author={Rathee, Mandeep and MacAvaney, Sean and Anand, Avishek},
  journal={arXiv preprint arXiv:2410.20286},
  year={2024}
}
```

