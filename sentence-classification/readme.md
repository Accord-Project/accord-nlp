# Sentence Classification 

This repository contains the single-clause classification research developed under the ACCORD NLP task. 

## Description 

This project involves a two-phase classification task. 

1. Phase 1: Clauses are identified as either _self-contained_, thus interpretable without external references, or not; _non-self-contained_ (Binary classification). This distinction is instrumental for the subsequent machine processing of these regulations.
2. Phase 2: The _self-contained clauses_ are further classified into _subjective_, _numerical_, or _combined_ (trinary classification) categories. This step is essential for practitioners, enabling quick identification of clauses suitable for direct automation and highlighting those requiring human intervention.

## Data

The binary classification task involves ~ 26k clauses available [here](https://github.com/Accord-Project/accord-nlp/blob/main/sentence-classification/data/Single-Clauses-Data_Binary-Classification.csv). However, the trinary classification task involves 1780 clauses available [here](https://github.com/Accord-Project/accord-nlp/blob/main/sentence-classification/data/Self-Contained-Clauses-Data_Trinary-Classification.csv).

## Results

For the binary classification task, we applied different Machine Learning techniques (Logistic Regression, Random Forest, SVM) with different NLP techniques for data representation, namely _tf.idf_ and _word2vec_. We also applied BERT as a deep learning technique.  The combination of the Random Forest classifier with TF-IDF features, assessed through 5-fold cross-validation, outperformed all other configurations, marking it as the most effective model for distinguishing between the two clauses categories as shown in the table below. Conversely, in the trinary classification scenario, BERT, with its profound contextual understanding achieved the highest accuracy, distinguishing itself as the premier model for classifying the self-contained clauses into three distinct categories: _numerical_, _subjective_, and _combined_. 

1. Phase 1: Binary classification: self-contained / non-self-contained

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Logistic Regression</th>
            <th><b>Random Forest</b></th>
            <th>Support VectorMachine</th>
            <th>BERT</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>TF.IDF</td>
            <td>97%</td>
            <td>99%</td>
            <td>99%</td>
            <td rowspan="3">95%</td>
        </tr>
        <tr>
            <td>TF.IDF (5-cross validation)</td>
            <td>96.6%</td>
            <td><b>99.3%</b></td>
            <td>99.1%</td>
        </tr>
        <tr>
            <td>word2vec</td>
            <td>87%</td>
            <td>98.9%</td>
            <td>95.6%</td>
        </tr>
    </tbody>
</table>

2. Phase 2: Trinary classification: numerical, subjective, combined
   <table>
    <thead>
        <tr>
            <th>Model</th>
            <th><b>Random Forest</b></th>
            <th>BERT</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>TF.IDF</td>
            <td>92%</td>
            <td rowspan="2">95%</td>
        </tr>
        <tr>
            <td>word2vec</td>
            <td>93%</td>
        </tr>
    </tbody>
</table>

## Implementation 

To run Python code for the two classification tasks using different classifiers and features, such as logistic regression, random forest, SVM, BERT, and feature representations like TF-IDF and Word2Vec, Python libraries that specialise in machine learning and natural language processing need to be installed, essentially: 

* **Scikit-learn**: For traditional machine learning classifiers like logistic regression, random forest, and SVM, and for TF-IDF features.
  ##### Installation:
   ```
   pip install scikit-learn
   ```
* **Gensim**: For Word2Vec feature extraction.
    ##### Installation:
   ```
   pip install gensim
   ```
* **Transformers**: For using BERT and other transformer models.
  ##### Installation:
   ```
   pip install transformers
   ```
### Code
The code for both classification phases is available [here](https://github.com/Accord-Project/accord-nlp/tree/main/sentence-classification/Code).

## Limitations

* Only single clauses were considered.
* Manual annotation is required to create the groundtruth, which is costly in term of time and resources.

