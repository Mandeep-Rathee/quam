# QuAM: Adaptive Retrieval through Query Affinity Modelling



# We are continuously polishing the code.

<p align="center">
  <img src="quam_main_fig.jpg" />
</p>

## Corpus Graph

We would like to thank the author of the GAR paper for providing the corpus graph. 
We use the same corpus graph for our experiments. 
We start with the 128 neighbours corpus graph (G_c) and create an affinity Graph (G_a).
For instance, the bm25 based corpus graph can be downloaded using:
```
graph = CorpusGraph.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128')
```


## Training Data for Learnt Affinity Model
The training data can be created using the laff.py file. Alternatively, we release the training dataset for learnt affinity model [here](https://zenodo.org/records/13363455) using Zenodo anonymously. The dataset has following files:

1. data-00000-of-00001.arrow
2. dataset_info.json
3. state.json

Please download all 3 files in the data/laff_train_data folder. Further, the dataset can be loaded as

```
import datasets
ds = datasets.load_from_disk("data/laff_train_data")
```

## Learnt Affinity Model
The Learnt affinity model can be trained using the train_laff.py file. Alternatively, we have released the model's weights anonymously using Zenodo and can be downloaded from [here](https://zenodo.org/records/13363455). The corresponding file is:

- bert-base-laff.pth

Please download the model at laff_model/ folder. Further the model can be loaded as follows:

```
from transformers import BertTokenizer, BertForSequenceClassification
from base_models import  BinaryClassificationBertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model_name = "bert-base-uncased" 
tokenizer = BertTokenizer.from_pretrained(base_model_name, torch_dtype=torch.float16)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=1,torch_dtype=torch.float16)

model = BinaryClassificationBertModel(base_model)
model.load_state_dict(torch.load(f"laff_model/bert-base-laff.pth"))
model.to(device)
```




