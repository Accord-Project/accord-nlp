# Created by Hansi at 24/07/2023
from experiments.ner.ner_config import ner_args
from experiments.ner.ner_cv_experiments import MODEL_TYPE
from text_classification.ner.ner_model import NERModel


model = NERModel(MODEL_TYPE, ner_args['best_model_dir'], args=ner_args)

input_text = ''
while input_text != 'exit':
    input_text = input('Enter the file name: ')
    predictions, raw_outputs = model.predict([input_text])
    print(predictions)

