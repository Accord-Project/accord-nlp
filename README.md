# ACCORD NLP Framework

This repository contains the NLP framework developed by ACCORD project. 

The optimisation of Automated Compliance Checking (ACC) within the Architecture, Engineering, and Construction (AEC) 
sector necessitates the interpretation of building codes, regulations, and standards into machine-processable formats. 
As these codes primarily exist in textual form, Natural Language Processing (NLP) is integral to decode this data while 
capturing the underlying linguistics and domain-specific characteristics.

## Index
1. [Installation](#installation)

# Installation <a name="installation"> </a>

As the initial step, Pytorch needs to be installed. The recommended Pytorch version is 2.0.1. Please refer to [PyTorch](https://pytorch.org/get-started/locally/#start-locally) 
installation page for the specific installation command for your platform.

Once PyTorch has been installed, accord-nlp can be installed either from the source or as a Python package via pip. 
The latter approach is recommended. 

## From Source
```
git clone https://github.com/Accord-Project/accord-nlp.git
cd accord-nlp
pip install -r requirements.txt
```

## From pip
```
pip install accord-nlp
```

# Features
1. Data Augmentation
2. Entity Classification
3. Relation Classification
4. Information Extraction

## Data Augmentation

Data augmentation supports the synthetic oversampling of relation annotated data within a domain-specific context. It 
can be used using the following code. The original experiment script is available [here]().

```python
from accord_nlp.data_augmentation import RelationDA

entities = ['object', 'property', 'quality', 'value']
rda = RelationDA(entity_categories=entities)

relations_path = '<.csv file path to original relation-annotated data>'
entities_path = '<.csv file path to entity samples per category>'
output_path = '<.csv file path to save newly created data>'
rda.replace_entities(relations_path, entities_path, output_path, n=12)
```

### Available Datasets

The data augmentation approach was applied to the relation-annotated training data in the CODE-ACCORD corpus. 2,912 
synthetic data samples were generated, resulting in a training set of 6,375 relations. More details about the data 
statistics are available in our paper.

The augmented training dataset can be loaded into a Pandas DataFrame using the following code.

```python
from datasets import Dataset
from datasets import load_dataset

data_files = {"augmented_train": "augmented.csv"}
augmented_train = Dataset.to_pandas(load_dataset("ACCORD-NLP/CODE-ACCORD-Relations", data_files=data_files, split="augmented_train"))
```

## Entity Classification

The entity classification problem is formulated as a sequence labelling problem and fine-tuned the following transformer-based 
architecture to build entity classifiers. 

The table below summarises the pre-trained transformer models that were experimented with and their performance details. 
Our paper provides more details about the experiments and evaluations conducted. 

<table>
    <thead>
        <tr>
            <th>Transformer</th>
            <th>Validation F1</th>
            <th>Test F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>BERT</td>
            <td>0.5686</td>
            <td>0.3649</td>
        </tr>
        <tr>
            <td>ELECTRA</td>
            <td>0.5964</td>
            <td>0.3059</td>
        </tr>
        <tr>
            <td>ALBERT</td>
            <td>0.6176</td>
            <td>0.3791</td>
        </tr>
        <tr>
            <td>ROBERTA</td>
            <td><b>0.6400</b></td>
            <td><b>0.3922</b></td>
        </tr>
        <tr>
            <td colspan="3"><center><i>Model Optimisation</i></center></td>
        </tr>
        <tr>
            <td>ROBERTA</td>
            <td><b>0.7351</b></td>
            <td><b>0.4444</b></td>
        </tr>
    </tbody>
</table>


### Available Models

All the fine-tuned models are available in [HuggingFace](https://huggingface.co/ACCORD-NLP), and can be accessed using the following code.

```python
from accord_nlp.text_classification.ner.ner_model import NERModel

model = NERModel('roberta', 'ACCORD-NLP/ner-roberta-large')
predictions, raw_outputs = model.predict(['The gradient of the passageway should not exceed five per cent.'])
print(predictions)
```

## Relation Classification

The following transformer-based architecture is adapted for relation classification. Four additional tokens: <e1>, </e1>, 
<e2> and </e2> are involved to format the model's input.

The table below summarises the pre-trained transformer models that we experimented with and their performance details. 
Our paper provides more details about the experiments and evaluations conducted.

<table>
    <thead>
        <tr>
            <th>Transformer</th>
            <th>Validation F1</th>
            <th>Test F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>BERT</td>
            <td>0.5144</td>
            <td>0.5243</td>
        </tr>
        <tr>
            <td>ALBERT</td>
            <td>0.5590</td>
            <td>0.5122</td>
        </tr>
        <tr>
            <td>ROBERTA</td>
            <td><b>0.5903</b></td>
            <td><b>0.5498</b></td>
        </tr>
        <tr>
            <td colspan="3"><center><i>Data Augmentation</i></center></td>
        </tr>
        <tr>
            <td>BERT</td>
            <td>0.9207</td>
            <td>0.8009</td>
        </tr>
        <tr>
            <td>ALBERT</td>
            <td>0.9109</td>
            <td>0.7556</td>
        </tr>
        <tr>
            <td>ROBERTA</td>
            <td><b>0.9450</b></td>
            <td><b>0.8011</b></td>
        </tr>
        <tr>
            <td colspan="3"><center><i>Model Optimisation</i></center></td>
        </tr>
        <tr>
            <td>ROBERTA</td>
            <td><b>0.9450</b></td>
            <td><b>0.8011</b></td>
        </tr>
    </tbody>
</table>

### Available Models

All the fine-tuned models are available in [HuggingFace](https://huggingface.co/ACCORD-NLP), and can be accessed using the following code.

```python
from accord_nlp.text_classification.relation_extraction.re_model import REModel

model = REModel('roberta', 'ACCORD-NLP/re-roberta-large')
predictions, raw_outputs = model.predict(['The <e1>gradient<\e1> of the passageway should not exceed <e2>five per cent</e2>.'])
print(predictions)
```

## Information Extraction

The Information Extraction(IE) pipeline comprises four integral components: (1) entity classifier, (2) entity pairer, 
(3) relation classifier and (4) graph builder, as illustrated in the following figure. Overall, when provided with a 
regulatory sentence as input, the IE pipeline produces a machine-readable output (i.e. an entity-relation graph) that 
encapsulates the information expressed in natural language. Our paper provides more details about each of the components. 

The default pipeline configurations are set to the best-performed entity and relation classification models, and the 
default pipeline can be accessed using the following code.

```python
from accord_nlp.information_extraction.ie_pipeline import InformationExtractor

sentence = 'The gradient of the passageway should not exceed five per cent.'

ie = InformationExtractor()
ie.sentence_to_graph(sentence)
```

The following code can be used to access the pipeline with different configurations. Please refer to the [ie_pipeline.py]() 
for more details about the input parameters. 

```python
from accord_nlp.information_extraction.ie_pipeline import InformationExtractor

sentence = 'The gradient of the passageway should not exceed five per cent.'

ner_model_info = ('roberta', 'ACCORD-NLP/ner-roberta-large')
re_model_info = ('roberta', 'ACCORD-NLP/re-roberta-large')
ie = InformationExtractor(ner_model_info=ner_model_info, re_model_info=re_model_info, debug=True)
ie.sentence_to_graph(sentence)

```




