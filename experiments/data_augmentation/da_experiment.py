# Created by Hansi at 23/08/2023
from accord_nlp.data_augmentation import RelationDA


entities = ['object', 'property', 'quality', 'value']

relations_path = '../../data/re/generated_train/greater-equal.csv'
entities_path = '../../data/re/generated_train/entities.csv'
output_path = '../../data/re/generated_train/synthetic/greater-equal.csv'

rda = RelationDA(entity_categories=entities)
rda.replace_entities(relations_path, entities_path, output_path, n=12)
