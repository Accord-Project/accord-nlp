# RASE to JSON 

This repository contains the RASE-to-JSON research developed under the ACCORD NLP task. 

## Description 

This project leverages Large Language Models (LLMs), specifically GPT-3.5, to automatically generate rules in JSON format. It does so by processing text from building regulations that have been tagged with the RASE methodology. This is accomplished by leveraging both long and short prompts. For long prompts, the entire HTML document marked with RASE tags is used, allowing the model to understand and generate rules in the context of the full document. However, short prompts focus on individual RASE-tagged blocks, enabling precise rule generation from specific sections. The approach combines few-shot prompting and fine-tuning techniques to efficiently translate the regulatory text into structured rules.

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


## Implementation 

**Step1: Finetuning**

Head over to [this code]()

Re-run from section __Dataset For Finetuning__ to __Creating Finetuning Job__ , a finetuned model id will be generatd

**Step2: Testing**

#### To Interact directly with the model, below is a sample function.

**The generated model id is required a sample is=> ft:gpt-3.5-turbo-0613:personal:regugen:8TyfLt5J**

OPENAI **api key** is required for every prompts

``` python code
def prompt_fm(NLT_Rules):
    # function that submit prompts to model and returns a json rule
    fine_tuned_model_id = "ft:gpt-3.5-turbo-0613:personal:regugen:8TyfLt5J"
    openai.api_key= <OPENAIKEY>
    response = openai.ChatCompletion.create(
    model=fine_tuned_model_id, 
    messages=[
        {"role": "system", "content": "You are an assistant for JSON Generation of Rules"},
        {"role": "user", "content": NLT_Rules}]
    , temperature=0
    )
    fm_res=response["choices"][0]["message"]["content"]
    print(fm_res)
    return fm_res;
```

**Where NLT_Rules is the extracted Natural Language Text from the HTML RASE tagged document**

**Sample Preprocessing**

``` python code
 NLT_Rules = extractNLTRules(RASETAGGED_Rules)
```

``` python code
output_data = StringIO()

def replace_whitespace(text):
    # A function that replace consecutive whitespaces with a single space
    return re.sub(r'\s+', ' ', text)


def extract_requirement_sections(tag):
    #
    #  A recursive function that extract requirements from each section
    # Check if the current tag is a Tag and has the "data-rasetype" attribute with the value "RequirementSection"
    #
    if isinstance(tag, Tag) and tag.get("data-rasetype") == "RequirementSection":

        # If found, store the content of the tag
        output_data.write("REQ: "+ replace_whitespace(tag.get_text())+ "\n")
    elif isinstance(tag, Tag):
        # If the current tag is a Tag, recursively call the function for each child of the current tag
        for child in tag.children:
            if(child.name != 'section' ):
                extract_requirement_sections(child)


def extract_section_info(section, d, soup):
    # A recursive function that extract Applications and Requirements(rules) from each sections keeping the hirachy
    baseSec = soup.find('section', {'title': section.get('title')})

    if(d==1):
        parghs = baseSec.find_all('p', recursive=False)
        for pargh in parghs:
            extract_requirement_sections(pargh)

    # check content
    subSecs= baseSec.find_all('section', recursive=False)
    for sec in subSecs:
        output_data.write("\t"*d +sec.get('title')+":"+ "\n")
        # extract_cont(sec,d, True)
        extract_requirement_sections(sec)

        extract_section_info(sec,d+1, soup)

def extractNLTRules(html_content) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    # Finding all the first node section tags to get sections from the html input
    sections = soup.find_all('section', limit=1)
    # Going through each of these sections to extract the rase tag applocations and rules/requirements
    for section in sections:
        output_data.write(section.get('title')+":"+ "\n")
        extract_section_info(section,1,soup)

    # Get the captured output as a string
    output_data.seek(0)
    output = output_data.read()
    # print("output:", output)
    # Closing StringIO
    output_data.truncate(0)
    output_data.seek(0)
    return output

```
__NB: After further investigation, it was discovered that if NLT is sent directly to the model it still generalised well by responding with a reasonable JSON rule__


### Code
The code for all experiments and related outputs is available [here]([https://github.com/Accord-Project/accord-nlp/tree/main/sentence-classification/Code](https://github.com/Falu-G/Regu-Genius/tree/main).

## Limitations

* Relatively small dataset for fine-tuning the LLM.
* Manual efforts to generate the JSON files from html files as a groundtruth.

