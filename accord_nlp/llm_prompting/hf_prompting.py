# Created by Hansi on 24/11/2023
import logging

import torch

# from langchain import HuggingFaceHub
# from langchain import HuggingFacePipeline
# from langchain import PromptTemplate, LLMChain

from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoTokenizer, pipeline


logger = logging.getLogger(__name__)

'''
supported models:
model_type = 'pipeline' > Falcon, Falcon-Instruct, Llama-2 (load the model to local machine)
model_type = 'hub' > Falcon-Instruct, flan-t5 (query via inference API, thus, can only handle small models with free version)
'''


class QAModel:
    def __init__(
            self,
            model_id,
            model_type,  # pipeline, hub
            max_length=200,
            temperature=0
    ):

        if model_type == 'pipeline':
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            _pipeline = pipeline(
                "text-generation",  # task
                model=model_id,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_length,
            )

            self.llm = HuggingFacePipeline(pipeline=_pipeline, model_kwargs={'temperature': temperature})

        elif model_type == 'hub':
            self.llm = HuggingFaceHub(repo_id=model_id,
                                      model_kwargs={"temperature": temperature, "max_length": max_length})

        else:
            raise ValueError("Unknown model_type found!")

    def query(self, template, question, samples):
        '''
        Query the model for each provided sample

        :param template: str
        :param question: str
        :param samples: str list
        :return: str list
            responses received from the model
        '''
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        responses = []

        for sample in samples:
            _question = f'{question} {sample}'
            response = llm_chain.run(question=_question)
            responses.append(response)
        return responses
