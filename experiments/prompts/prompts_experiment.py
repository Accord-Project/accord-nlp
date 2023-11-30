# Created by Hansi on 30/11/2023
import argparse
import os

from accord_nlp.llm_prompting.hf_prompting import QAModel
from accord_nlp.llm_prompting.templates import zero_shot_hate_speech_template, zero_shot_hate_speech_question

parser = argparse.ArgumentParser(description='''LLM-based prompting  ''')
parser.add_argument('--model_id', required=False, help='model name', default="tiiuae/falcon-7b")
parser.add_argument('--model_type', required=False, help='model type', default="hub")
parser.add_argument('--max_length', required=False, help='max sequence length', default=200)
parser.add_argument('--temperature', required=False, help='value to modulate the next token probabilities', default=0)
parser.add_argument('--huggingfacehub_api_token', required=False, help='HuggingFaceHub api token', default=None)
arguments = parser.parse_args()

model_id = arguments.model_id
model_type = arguments.model_type
max_length = arguments.max_length
temperature = arguments.temperature

# huggingfacehub_api_token is only required if model_type='hub'
if arguments.huggingfacehub_api_token is not None:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = arguments.huggingfacehub_api_token


model = QAModel(model_id, model_type, max_length, temperature)

samples = ["5 Tips to Enhance Audience Connection on Facebook URL @USER #socialmedia #smm URL",
           "#StopKavanaugh he is liar like the rest of the #GOP URL"]
responses = model.query(zero_shot_hate_speech_template, zero_shot_hate_speech_question, samples)
print(responses)

