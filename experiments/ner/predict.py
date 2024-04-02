# Created by Hansi on 25/03/2024
from accord_nlp.text_classification.ner.ner_model import NERModel

model = NERModel('roberta', 'ACCORD-NLP/ner-roberta-large')
predictions, raw_outputs = model.predict(['The gradient of the passageway should not exceed five per cent.'])
print(predictions)
