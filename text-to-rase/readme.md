# Text To RASE 

This repository contains the text-to-RASE research developed under the ACCORD NLP task. 

## Description 

This project treats the RASE automation process as a nested NER task using various NLP techniques, namely SpanBERT, FLAIR and T5 (Text-To-Text-Transfer-transformer).

## Data

 This task used the RASE-tagged html documents as a primary data source.

## Results

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>spanBERT</td>
            <td>72.74%</td>
        </tr>
        <tr>
            <td>FLAIR</td>
            <td>84.57%</td>
        </tr>
        <tr>
            <td>T5</td>
            <td>97.12%</td>
        </tr>
    </tbody>
</table>

### Code
* **spanBERT**: [code](https://github.com/vinnyonodu/Auto-RASE_txt_tagging_SpanBERT_research)
* **FLAIR**: [code](https://github.com/vinnyonodu/Auto-RASE_txt_tagging_flair_research)
* **T5**: [code](https://github.com/vinnyonodu/Auto-RASE_txt_tagging_t5_research)
* **User Interface**: [code](https://github.com/vinnyonodu/Auto-RASE_txt_tagging_user_interface)

## Limitations

* This research does not capture the hierarchical structure of RASE.
