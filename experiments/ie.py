# Created by Hansi on 26/03/2024
from accord_nlp.information_extraction.ie_pipeline import InformationExtractor

if __name__ == '__main__':
    # sentence = 'Perimeter insulation should be continuous and have a minimum thickness of 25mm.'
    # sentence = 'The access route for pedestrians/wheelchair users shall not be steeper than 1:20.'
    sentence = 'The gradient of the passageway should not exceed five per cent.'
    ie = InformationExtractor()
    ie.sentence_to_graph(sentence)

