# Created by Hansi at 28/08/2023

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

import torch
from nltk import word_tokenize

from accord_nlp.information_extraction.convertor import entity_pairing, graph_building
from accord_nlp.text_classification.ner.ner_model import NERModel
from accord_nlp.text_classification.relation_extraction.re_model import REModel

SEED = 157

ner_args = {
    "labels_list": ["O", "B-quality", "B-property", "I-property", "I-quality", "B-object", "I-object", "B-value", "I-value"],
}

re_args = {
    "labels_list": ["selection", "necessity", "none", "greater", "part-of", "equal", "greater-equal", "less-equal", "not-part-of", "less"],
    "special_tags": ["<e1>", "<e2>"],  # Should be either begin_tag or end_tag
}


class InformationExtractor:
    def __init__(
            self,
            ner_model_info=('roberta', 'ACCORD-NLP/ner-roberta-large', ner_args),
            re_model_info=('roberta', 'ACCORD-NLP/re-roberta-large', re_args),
            cuda_device=0):

        self.ner_model = NERModel(ner_model_info[0], ner_model_info[1], labels=ner_model_info[2]['labels_list'],
                                  use_cuda=torch.cuda.is_available(), cuda_device=cuda_device, args=ner_model_info[2])

        self.re_model = REModel(re_model_info[0], re_model_info[1], use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device, args=re_model_info[2])

    def preprocess(self, sentence):
        # remove white spaces at the beginning and end of the text
        sentence = sentence.strip()
        # tokenise
        sentence = ' '.join(word_tokenize(sentence))
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

        # pair entities to predict their relations
        entity_pair_df = entity_pairing(sentence, ner_predictions[0])

        # relation extraction
        re_predictions, re_raw_outputs = self.re_model.predict(entity_pair_df['output'].tolist())
        entity_pair_df['prediction'] = re_predictions

        # build graph
        graph = graph_building(entity_pair_df, view=False)

        return graph


# if __name__ == '__main__':
#     sentence = 'Perimeter insulation should be continuous and have a minimum thickness of 25mm.'
#     ie = InformationExtractor()
#     ie.sentence_to_graph(sentence)
