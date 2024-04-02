# Created by Hansi at 23/08/2023
import random
import re

import pandas as pd
from tqdm import tqdm


class RelationDA:
    def __init__(self, entity_categories):
        """
        Initialise RelationDA object

        :param entity_categories: list of str
            unique entity categories found in the data
        """
        self.entities = entity_categories

    def replace_entities(self, relations_path, entities_path, output_path, n=10):
        """
        Augment data by replacing entities

        :param relations_path: str
            .csv of original relation data
            required columns: example_id, tagged_sentence, relation_type, e1_tag, e2_tag
        :param entities_path: str
            .csv of entity samples per category
        :param output_path: str
            .csv to save newly created data
        :param n: int, optional
            augmentation factor - number of new samples to be created using an original sample
        :return:
        """
        relations_df = pd.read_csv(relations_path, encoding='utf-8')
        relations_df.dropna(inplace=True)
        entities_df = pd.read_csv(entities_path, encoding='utf-8')

        entities_dict = {}
        for entity in self.entities:
            entities_dict[entity] = entities_df[entity].dropna().tolist()

        new_data = []

        for index, row in tqdm(relations_df.iterrows()):
            id = row['example_id']
            # print(id)
            tagged_sentence = row['tagged_sentence']
            relation = row['relation_type']
            e1_text = re.search('<e1>(.*)</e1>', tagged_sentence).group(1)
            e2_text = re.search('<e2>(.*)</e2>', tagged_sentence).group(1)

            e1_tag = row['e1_tag']
            e2_tag = row['e2_tag']

            if e1_tag == e2_tag:
                temp_entities = entities_dict[e1_tag].copy()
                if e1_text in temp_entities:
                    temp_entities.remove(e1_text)
                if e2_text in temp_entities:
                    temp_entities.remove(e2_text)
                replacements = random.sample(temp_entities, 2 * n)
                e1_replacements = replacements[:n]
                e2_replacements = replacements[n:]

            else:
                e1_temp_entities = entities_dict[e1_tag].copy()
                if e1_text in e1_temp_entities:
                    e1_temp_entities.remove(e1_text)
                e1_replacements = random.sample(e1_temp_entities, n)

                e2_temp_entities = entities_dict[e2_tag].copy()
                if e2_text in e2_temp_entities:
                    e2_temp_entities.remove(e2_text)
                e2_replacements = random.sample(e2_temp_entities, n)

            i = 0
            for e1_replacement, e2_replacement in zip(e1_replacements, e2_replacements):
                replaced = re.sub('<e1>(.*)</e1>', f'<e1>{e1_replacement}</e1>', tagged_sentence)
                replaced = re.sub('<e2>(.*)</e2>', f'<e2>{e2_replacement}</e2>', replaced)

                new_data.append([f'{id}_{i}', replaced, relation])
                i = i + 1

        output_df = pd.DataFrame(new_data, columns=['example_id', 'tagged_sentence', 'relation_type'])
        output_df.to_csv(output_path, encoding='utf-8', index=False)
