# ACCORD-NLP Framework

ACCORD-NLP is a Natural Language Processing (NLP) framework developed as a part of the Horizon European project for  Automated Compliance Checks for Construction, Renovation or Demolition Works ([ACCORD](https://accordproject.eu/)) to facilitate Automated Compliance Checking (ACC) within the Architecture, Engineering, and Construction (AEC) sector.

Compliance checking plays a pivotal role in the AEC sector, ensuring the safety, reliability, stability, and usability of building designs. Traditionally, this process relied on manual approaches, which are resource-intensive and time-consuming. Thus, attention has shifted towards automated methods to streamline compliance checks. Automating these processes necessitates the transformation of building regulations written in text aiming domain experts into machine-processable formats. However, this has been challenging, primarily due to the inherent complexities and unstructured nature of natural languages. Moreover, regulatory texts often exhibit domain-specific characteristics, ambiguities, and intricate clausal structures, further complicating the task.

ACCORD-NLP offers data, AI models and workflows developed using state-of-the-art NLP techniques to extract rules from textual data, supporting ACC.

## Installation <a name="installation"> </a>

As the initial step, Pytorch needs to be installed. The recommended Pytorch version is 2.0.1. Please refer to [PyTorch](https://pytorch.org/get-started/locally/#start-locally) 
installation page for the specific installation command for your platform.

Once PyTorch has been installed, accord-nlp can be installed either from the source or as a Python package via pip. 
The latter approach is recommended. 

### From Source
```
git clone https://github.com/Accord-Project/accord-nlp.git
cd accord-nlp
pip install -r requirements.txt
```

### From pip
```
pip install accord-nlp
```

## Features
1. [Data Augmentation](#da)
2. [Entity Classification](#ner)
3. [Relation Classification](#re)
4. [Information Extraction](#ie)

### Data Augmentation <a name="da"> </a>

Data augmentation supports the synthetic oversampling of relation-annotated data within a domain-specific context. It can be used using the following code. The original experiment script is available [here](https://github.com/Accord-Project/accord-nlp/blob/main/experiments/data_augmentation/da_experiment.py).

```python
from accord_nlp.data_augmentation import RelationDA

entities = ['object', 'property', 'quality', 'value']
rda = RelationDA(entity_categories=entities)

relations_path = '<.csv file path to original relation-annotated data>'
entities_path = '<.csv file path to entity samples per category>'
output_path = '<.csv file path to save newly created data>'
rda.replace_entities(relations_path, entities_path, output_path, n=12)
```

#### Available Datasets

The data augmentation approach was applied to the relation-annotated training data in the [CODE-ACCORD](https://github.com/Accord-Project/CODE-ACCORD) corpus. It generated 2,912 synthetic data samples, resulting in a training set of 6,375 relations. Our paper, listed below, provides more details about the data statistics.

The augmented training dataset can be loaded into a Pandas DataFrame using the following code.

```python
from datasets import Dataset
from datasets import load_dataset

data_files = {"augmented_train": "augmented.csv"}
augmented_train = Dataset.to_pandas(load_dataset("ACCORD-NLP/CODE-ACCORD-Relations", data_files=data_files, split="augmented_train"))
```

### Entity Classification <a name="ner"> </a>

We adapted the transformer's sequence labelling architecture to fine-tune the entity classifier, following its remarkable results in the NLP domain. The general transformer architecture was modified by adding individual softmax layers per output token to support entity classification. 

Our paper, listed below, provides more details about the model architecture, fine-tuning process, experiments and evaluations. 

#### Available Models

We fine-tuned four pre-trained transformer models (i.e. BERT, ELECTRA, ALBERT and ROBERTA) for entity classification.
All the fine-tuned models are available in [HuggingFace](https://huggingface.co/ACCORD-NLP), and can be accessed using the following code.

```python
from accord_nlp.text_classification.ner.ner_model import NERModel

model = NERModel('roberta', 'ACCORD-NLP/ner-roberta-large')
predictions, raw_outputs = model.predict(['The gradient of the passageway should not exceed five per cent.'])
print(predictions)
```

### Relation Classification <a name="re"> </a>

Relation classification aims to predict the semantic relationship between two entities within a context. We introduced four special tokens (i.e. \<e1>, \</e1>, \<e2> and \</e2>) to format the input text with an entity pair to facilitate relation classification. Both \<e1> and \</e1> mark the start and end of the first entity in the selected text sequence, while \<e2> and \</e2> mark the start and end of the second entity. The transformer output corresponds to \<e1> and \<e2> were passed through a softmax layer to predict the relation category. 

Our paper, listed below, provides more details about the model architecture, fine-tuning process, experiments and evaluations.

#### Available Models

We fine-tuned three pre-trained transformer models (i.e. BERT, ALBERT and ROBERTA) for relation classification. 
All the fine-tuned models are available in [HuggingFace](https://huggingface.co/ACCORD-NLP), and can be accessed using the following code.

```python
from accord_nlp.text_classification.relation_extraction.re_model import REModel

model = REModel('roberta', 'ACCORD-NLP/re-roberta-large')
predictions, raw_outputs = model.predict(['The <e1>gradient<\e1> of the passageway should not exceed <e2>five per cent</e2>.'])
print(predictions)
```

### Information Extraction <a name="ie"> </a>

Our information extraction pipeline aims to transform a regulatory sentence into a machine-processable output (i.e., a knowledge graph of entities and relations). It utilises the entity and relation classifiers mentioned above to sequentially extract information from the text to build the final graph. 

Our paper, listed below, provides more details about the pipeline, including its individual components. 


The default pipeline configurations are set to the best-performed entity and relation classification models, and the 
default pipeline can be accessed using the following code.

```python
from accord_nlp.information_extraction.ie_pipeline import InformationExtractor

sentence = 'The gradient of the passageway should not exceed five per cent.'

ie = InformationExtractor()
ie.sentence_to_graph(sentence)
```

The following code can be used to access the pipeline with different configurations. Please refer to the [ie_pipeline.py](https://github.com/Accord-Project/accord-nlp/blob/main/accord_nlp/information_extraction/ie_pipeline.py) 
for more details about the input parameters. 

```python
from accord_nlp.information_extraction.ie_pipeline import InformationExtractor

sentence = 'The gradient of the passageway should not exceed five per cent.'

ner_model_info = ('roberta', 'ACCORD-NLP/ner-roberta-large')
re_model_info = ('roberta', 'ACCORD-NLP/re-roberta-large')
ie = InformationExtractor(ner_model_info=ner_model_info, re_model_info=re_model_info, debug=True)
ie.sentence_to_graph(sentence)

```

Also, a live demo of the Information Extractor is available in [HuggingFace](https://huggingface.co/spaces/ACCORD-NLP/information-extractor). 


## Reference

*Please note that the corresponding paper for this work is currently in progress and will be made available soon. Thank you for your patience and interest.*



