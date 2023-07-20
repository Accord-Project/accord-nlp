# Created by Hansi at 19/07/2023
import re
from itertools import combinations

import numpy as np
import pandas as pd


def validate_indices(input_path, output_path):
    df = pd.read_excel(input_path, sheet_name='Relations')
    validation_e1 = []
    validation_e2 = []

    for index, row in df.iterrows():
        e1 = (row['content'])[int(row['entity1_start']): int(row['entity1_end'])]
        v1 = 1 if e1 == str(row['entity1_value']) else 0
        validation_e1.append(v1)

        e2 = (row['content'])[int(row['entity2_start']): int(row['entity2_end'])]
        v2 = 1 if e2 == str(row['entity2_value']) else 0
        validation_e2.append(v2)

    df['validation_e1'] = validation_e1
    df['validation_e2'] = validation_e2
    df.to_csv(output_path, encoding='utf-8', index=False)


def tag_sentence(sentence, e1_start, e1_end, e2_start, e2_end, e1_tag=None, e2_tag=None):
    """
    Add entity tags to a sentence
    :param sentence: str
    :param e1_start: int
    :param e1_end: int
    :param e2_start: int
    :param e2_end: int
    :param e1_tag: str, optional
    :param e2_tag: str, optional
    :return: str
    """
    if e2_start < e1_start:
        e1_start, e1_end, e2_start, e2_end = e2_start, e2_end, e1_start, e1_end

    tagged_sentence = f"{sentence[:e1_start]}<e1>{sentence[e1_start:e1_end]}</e1>{sentence[e1_end:e2_start]}<e2>{sentence[e2_start:e2_end]}</e2>{sentence[e2_end:]}"
    if e1_tag is not None and e2_tag is not None:
        tagged_sentence = tagged_sentence.replace('e1', f"e1-{e1_tag}")
        tagged_sentence = tagged_sentence.replace('e2', f"e2-{e2_tag}")

    tokens = re.split(r'\s+', tagged_sentence)  # tokenise using spaces
    if tokens[0] == '': del tokens[0]
    if tokens[len(tokens) - 1] == '': del tokens[len(tokens) - 1]
    return ' '.join(tokens)


def tag_entities(input_path, output_path, add_entity_tag=False):
    """
    Format sentences to entity-tagged sentence
    :param input_path: .csv (final-relations)
        compulsory columns: example_id, content, metadata, entity1_start, entity1_end, entity1_tag, entity2_start,
        entity2_end, entity2_tag, relation_type
    :param output_path: .csv
    :param add_entity_tag: boolean, optional
    :return:
    """
    df = pd.read_csv(input_path, encoding='utf-8')

    if not add_entity_tag:
        df['tagged_sentence'] = df.apply(
            lambda row: tag_sentence(row['content'], row['entity1_start'], row['entity1_end'], row['entity2_start'],
                                     row['entity2_end']), axis=1)
    else:
        df['tagged_sentence'] = df.apply(
            lambda row: tag_sentence(row['content'], row['entity1_start'], row['entity1_end'], row['entity2_start'],
                                     row['entity2_end'], row['entity1_tag', row['entity2_tag']]), axis=1)

    df = df[['example_id', 'content', 'metadata', 'tagged_sentence', 'relation_type']]
    df.to_csv(output_path, encoding='utf-8', index=False)


def pair_entities(data_path, output_path=None):
    df = pd.read_csv(data_path, encoding='utf-8')

    ids = df['example_id'].unique().tolist()
    data_list = []
    for id in ids:
        filtered_df = df.loc[df['example_id'] == id]
        filtered_df.reset_index(drop=True, inplace=True)
        row0 = filtered_df.iloc[0]
        sentence = row0['content']
        metadata = row0['metadata']

        indices = list(filtered_df.index.values)
        pairs = list(combinations(indices, 2))
        print(f'{id}:{pairs}')

        for pair in pairs:
            r1 = filtered_df.iloc[pair[0]]
            r2 = filtered_df.iloc[pair[1]]
            if r1['start'] < r2['start']:
                data_list.append(
                    [id, sentence, metadata, r1['start'], r1['end'], r1['value'], r1['tag'], r2['start'], r2['end'],
                     r2['value'], r2['tag']])
            else:
                data_list.append(
                    [id, sentence, metadata, r2['start'], r2['end'], r2['value'], r2['tag'], r1['start'], r1['end'],
                     r1['value'], r1['tag']])

    output_df = pd.DataFrame(data_list, columns=['example_id', 'content', 'metadata', 'entity1_start', 'entity1_end',
                                                 'entity1_value', 'entity1_tag', 'entity2_start', 'entity2_end',
                                                 'entity2_value', 'entity2_tag'])
    if output_path is not None:
        output_df.to_csv(output_path, encoding='utf-8', index=False)
    return output_df


def get_non_related_pairs(entity_path, relation_path, output_path):
    relations_df = pd.read_csv(relation_path, encoding='utf-8')
    relations_df = relations_df[['example_id', 'content', 'metadata', 'entity1_start', 'entity1_end',
                                 'entity1_value', 'entity1_tag', 'entity2_start', 'entity2_end',
                                 'entity2_value', 'entity2_tag', 'relation_type']]
    # sort entities
    relations_df['entity1_start'], relations_df['entity1_end'], relations_df['entity1_value'], relations_df[
        'entity1_tag'], relations_df['entity2_start'], relations_df['entity2_end'], relations_df['entity2_value'], \
        relations_df['entity2_tag'] = np.where(relations_df['entity2_start'] < relations_df['entity1_start'],
                                               (relations_df['entity2_start'], relations_df['entity2_end'],
                                                relations_df['entity2_value'], relations_df[
                                                    'entity2_tag'], relations_df['entity1_start'],
                                                relations_df['entity1_end'], relations_df['entity1_value'], \
                                                relations_df['entity1_tag']), (
                                                   relations_df['entity1_start'], relations_df['entity1_end'],
                                                   relations_df['entity1_value'], relations_df[
                                                       'entity1_tag'], relations_df['entity2_start'],
                                                   relations_df['entity2_end'], relations_df['entity2_value'], \
                                                   relations_df['entity2_tag']))
    # relations_df.to_csv(output_path, encoding='utf-8', index=False)

    entity_pairs_df = pair_entities(entity_path)

    # merge two DataFrames and create indicator column
    df_all = entity_pairs_df.merge(relations_df.drop_duplicates(),
                                   on=['example_id', 'content', 'metadata', 'entity1_start', 'entity1_end',
                                       'entity1_value', 'entity1_tag', 'entity2_start', 'entity2_end',
                                       'entity2_value', 'entity2_tag'], how='left', indicator=True)

    # create DataFrame with rows that exist in first DataFrame only
    non_related_pairs_df = df_all[df_all['_merge'] == 'left_only']
    non_related_pairs_df.loc[:, 'relation_type'] = 'None'
    non_related_pairs_df = non_related_pairs_df.drop('_merge', axis=1)

    non_related_pairs_df.to_csv(output_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    input_path = '../../data/Final-Data.xlsx'
    output_path = '../../data/re/validated.csv'
    # validate_indices(input_path, output_path)

    input_path = '../../data/re/validated.csv'
    output_path = '../../data/re/all.csv'
    # tag_entities(input_path, output_path, add_entity_tag=False)

    data_path = '../../data/ner/validated-entities.csv'
    output_path = '../../data/re/all_pairs.csv'
    # pair_entities(data_path, output_path)

    entity_path = '../../data/ner/validated-entities.csv'
    relation_path = '../../data/re/validated.csv'
    output_path = '../../data/re/none_relations.csv'
    get_non_related_pairs(entity_path, relation_path, output_path)
