# Created by Hansi on 25/03/2024
from accord_nlp.text_classification.relation_extraction.re_model import REModel

model = REModel('roberta', 'ACCORD-NLP/re-roberta-large')
predictions, raw_outputs = model.predict(['The <e1>gradient<\e1> of the passageway should not exceed <e2>five per cent</e2>.'])
print(predictions)

