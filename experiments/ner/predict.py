# Created by Hansi at 24/07/2023
import argparse

from experiments.ner.ner_config import ner_args
from text_classification.ner.ner_model import NERModel

parser = argparse.ArgumentParser(description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name

model = NERModel(MODEL_TYPE, MODEL_NAME, args=ner_args)

input_text = ''
while input_text != 'exit':
    input_text = input('Enter the file name: ')
    predictions, raw_outputs = model.predict([input_text])
    print(predictions)

    final_predictions = []
    for prediction in predictions:
        raw_prediction = []
        for word_prediction in prediction:
            for key, value in word_prediction.items():
                raw_prediction.append(value)
        final_predictions.append(raw_prediction)
    print(' '.join(final_predictions[0]))
