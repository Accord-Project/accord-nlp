# Created by Hansi on 26/03/2024
from accord_nlp.information_extraction.ie_pipeline import InformationExtractor

if __name__ == '__main__':
    # sentence = 'Perimeter insulation should be continuous and have a minimum thickness of 25mm.'
    # sentence = 'The access route for pedestrians/wheelchair users shall not be steeper than 1:20.'
    sentence = 'The gradient of the passageway should not exceed five per cent.'
    # ie = InformationExtractor()
    # ie.sentence_to_graph(sentence)

    ner_model_info = ('roberta', 'ACCORD-NLP/ner-roberta-large')
    re_model_info = ('roberta', 'ACCORD-NLP/re-roberta-large')
    ie = InformationExtractor(ner_model_info=ner_model_info, re_model_info=re_model_info)
    ie.sentence_to_graph(sentence)

