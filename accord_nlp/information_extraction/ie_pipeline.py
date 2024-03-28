# Created by Hansi at 28/08/2023
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import logging

import torch
from nltk import word_tokenize

from accord_nlp.information_extraction.convertor import entity_pairing, graph_building
from accord_nlp.text_classification.ner.ner_model import NERModel
from accord_nlp.text_classification.relation_extraction.re_model import REModel

logger = logging.getLogger(__name__)


class InformationExtractor:
    def __init__(
            self,
            ner_model_info=('roberta', 'ACCORD-NLP/ner-roberta-large'),
            re_model_info=('roberta', 'ACCORD-NLP/re-roberta-large'),
            cuda_device=0,
            debug=False):

        """
        Initialise an information extraction pipeline

        :param ner_model_info: tuple
            (<model_type>, <model_name>) OR (<model_type>, <model_name>, <args as a dictionary>)
        :param re_model_info: tuple
            (<model_type>, <model_name>) OR (<model_type>, <model_name>, <args as a dictionary>)
        :param cuda_device: int (optional)
        :param debug: boolean (optional)
            If debug=True, intermediate outputs will be logged
        """

        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.INFO)

        self.ner_model = NERModel(ner_model_info[0], ner_model_info[1], use_cuda=torch.cuda.is_available(),
                                  cuda_device=cuda_device, args=ner_model_info[2] if len(ner_model_info) > 2 else None)

        self.re_model = REModel(re_model_info[0], re_model_info[1], use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device, args=re_model_info[2] if len(re_model_info) > 2 else None)

    def preprocess(self, sentence):
        sentence = sentence.strip()  # remove white spaces at the beginning and end of the text
        sentence = ' '.join(word_tokenize(sentence))  # tokenise the sentence
        return sentence

    def sentence_to_graph(self, sentence):
        """
        Generate a graph based on the information contained in a sentence
            graph nodes - entities
            graph edges - relations between entities

        :param sentence: str
        :return: graphviz graph
        """
        # preprocess
        sentence = self.preprocess(sentence)

        # NER
        ner_predictions, ner_raw_outputs = self.ner_model.predict([sentence])
        if self.debug:
            logger.info(f'Entity predictions: {ner_predictions}')

        # pair entities to predict their relations
        entity_pair_df = entity_pairing(sentence, ner_predictions[0])

        # relation extraction
        re_predictions, re_raw_outputs = self.re_model.predict(entity_pair_df['output'].tolist())
        entity_pair_df['prediction'] = re_predictions
        if self.debug:
            logger.info('Relation predictions:')
            for pred, sample in zip(re_predictions, entity_pair_df['output']):
                logger.info(f'{sample}: {pred}')

        # build graph
        graph = graph_building(entity_pair_df, view=True)

        return graph
