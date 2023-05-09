# Created by Hansi at 28/03/2023

import json
import os
from itertools import groupby

import numpy as np
import pandas as pd


class BasicRelation:
    def __init__(self, id, parend_span_id, child_span_id, relation_type, annotator_id):
        self.id = id
        self.parend_span_id = parend_span_id
        self.child_span_id = child_span_id
        self.relation_type = relation_type
        self.annotator_id = annotator_id


class Relation:
    def __init__(self, id, parent_span, child_span, relation_type, annotator_id, sample_id):
        self.id = id
        self.parent_span = parent_span
        self.child_span = child_span
        self.relation_type = relation_type
        self.annotator_id = annotator_id
        self.sample_id = sample_id
        self.vote = None

    def set_annotators(self, annotator_ids):
        self.annotator_id = annotator_ids

    def set_vote(self, vote):
        self.vote = vote

    def as_dict(self):
        """
        For CSV generation
        :return: dict
        """
        return {'relation_id': self.id, 'entity1_tagged_token_id': self.parent_span['tagged_token_id'],
                'entity1_start': self.parent_span['start'], 'entity1_end': self.parent_span['end'],
                'entity1_value': self.parent_span['value'], 'entity1_tag': self.parent_span['tag'],
                'entity2_tagged_token_id': self.child_span['tagged_token_id'],
                'entity2_start': self.child_span['start'],
                'entity2_end': self.child_span['end'], 'entity2_value': self.child_span['value'],
                'entity2_tag': self.child_span['tag'], 'relation_type': self.relation_type,
                'annotator_id': self.annotator_id, 'vote': self.vote, 'example_id': self.sample_id}


def relations_to_csv(list_relations, csv_file_path=None):
    df = pd.DataFrame([x.as_dict() for x in list_relations])
    if csv_file_path is not None:
        df.to_csv(csv_file_path, encoding='utf-8', index=False)
    return df


def get_relations(json_path):
    """
    Get relations in JSON as a list of BasicRelations

    :param json_path: str
    :return: list of BasicRelations
    """
    results = json.load(open(json_path))

    relations = results['relations']
    dict_relations = {}
    for r in relations:
        dict_relations[r['id']] = r

    list_relations = []
    for k, v in dict_relations.items():
        if v['relation_type'] is not None:
            parent = dict_relations[v['parent_id']]
            parent_span_id = parent['tagged_token_id']
            child_span_id = v['tagged_token_id']
            temp_relation = BasicRelation(k, parent_span_id, child_span_id, v['relation_type'], v['annotator_id'])
            list_relations.append(temp_relation)

    # Relation count print here will not be similar to number of elements in relations. This prints the two-node
    # relation count.
    print(f'total relations: {len(list_relations)}')
    return list_relations


def format_relations(json_path, list_basic_relations):
    """
    Format BasicRelations to Relations

    :param json_path: str
    :param list_basic_relations: list of BasicRelations
    :return: list of Relations
    """
    results = json.load(open(json_path))
    examples = results['examples']
    all_annotations = sum(map(lambda x: x['annotations'], examples), [])
    dict_all_annotations = {a['tagged_token_id']: a for a in all_annotations}

    list_relations = []

    for basic_relation in list_basic_relations:
        parent_span = dict_all_annotations[basic_relation.parend_span_id]
        child_span = dict_all_annotations[basic_relation.child_span_id]

        if parent_span["start"] > child_span["start"]:
            temp_parent_span = parent_span
            parent_span = child_span
            child_span = temp_parent_span

        sample_id = parent_span['example_id']
        relation = Relation(basic_relation.id, parent_span, child_span, basic_relation.relation_type,
                            basic_relation.annotator_id, sample_id)
        list_relations.append(relation)

    return list_relations


def process_relations(json_path, list_relations, final_annotation_path, output_folder_path, majority_vote=2,
                      filter=True):
    """
    Group, filter and make decisions on relations

    :param json_path: str
    :param list_relations: list of Relations
    :param final_annotation_path: str
        path to a .csv of processed entities with the compulsory columns: decision and tagged_token_id
    :param output_folder_path: str
    :param majority_vote: int, optional
        the majority vote to consider for finalised decisions
    :param filter: boolean, optional
        True: filter relations based on finalised entities and make relation decisions
        False: consider all entities and do not make relation decisions
    :return:
    """
    results = json.load(open(json_path))
    examples = results['examples']
    df_examples = pd.DataFrame(examples, columns=['example_id', 'content', 'metadata'])

    # get tagged_token_ids of finalised span annotations
    df_annotations = pd.read_csv(final_annotation_path)
    if filter:
        df_final_annotations = df_annotations[df_annotations['decision'] == 'finalised'].copy()
        tagged_token_ids = df_final_annotations['tagged_token_id'].to_list()

    keyfunc = lambda \
            r: f'{r.sample_id}_{r.parent_span["start"]}_{r.parent_span["end"]}_{r.parent_span["tag"]}_{r.relation_type}_{r.child_span["start"]}_{r.child_span["end"]}_{r.child_span["tag"]}'
    data = sorted(list_relations, key=keyfunc)

    groups = []  # keep groups
    grouped_relations = []  # keep combined groups by annotators
    filtered_grouped_relations = []  # keep combined groups which have finalised tags

    for k, g in groupby(data, keyfunc):
        group_elements = list(g)
        # update relation with annotators
        annotators = set([e.annotator_id for e in group_elements])
        print(f'{k}: {annotators}')

        grouped_relation = group_elements[0]

        grouped_relation.set_annotators(annotators)
        grouped_relation.set_vote(len(annotators))

        if filter:
            # filter relations based on finalised spans
            if grouped_relation.parent_span['tagged_token_id'] in tagged_token_ids and grouped_relation.child_span['tagged_token_id'] in tagged_token_ids:
                filtered_grouped_relations.append(grouped_relation)

        grouped_relations.append(grouped_relation)
        groups.append(group_elements)

    df_relations = relations_to_csv(grouped_relations)
    df_relations_final = pd.merge(df_examples, df_relations, on='example_id')
    df_relations_final.sort_values(by=['example_id', 'entity1_start', 'entity2_start'], inplace=True, ignore_index=True)
    df_relations_final.to_csv(os.path.join(output_folder_path, 'relations.csv'), encoding='utf-8', index=False)

    if filter:
        df_filtered_relations = relations_to_csv(filtered_grouped_relations)
        df_filtered_relations['decision'] = np.where(df_filtered_relations['vote'] >= majority_vote, 'finalised',
                                                     'pending')

        df_filtered_relations_final = pd.merge(df_examples, df_filtered_relations, on='example_id')
        df_filtered_relations_final.sort_values(by=['example_id', 'entity1_start', 'entity2_start'], inplace=True,
                                                ignore_index=True)
        df_filtered_relations_final.to_csv(os.path.join(output_folder_path, 'final-relations.csv'), encoding='utf-8',
                                           index=False)


if __name__ == '__main__':
    json_path = 'data/round2/accord-r2-b_annotations.json'
    final_annotation_path = 'data/round2/accord-r1-b-tags.csv'
    output_folder_path = 'data/round2/'

    list_basic_relations = get_relations(json_path)
    list_relations = format_relations(json_path, list_basic_relations)

    process_relations(json_path, list_relations, final_annotation_path, output_folder_path, majority_vote=2,
                      filter=False)
