# Created by Hansi at 28/03/2023

import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from process_relations import get_relations, format_relations


def plot_matrix(matrix, labels, filename=None):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    matrix = np.round(matrix, decimals=4)

    ax.matshow(matrix, cmap=plt.cm.Blues, vmin=-1, vmax=1)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            c = matrix[i, j]
            ax.text(j, i, str(c), va='center', ha='center')

    if filename is not None:
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')


def pairwise_agreement(annotator_ids, dict_annotator_x_ids):
    """
    Calculate pairwise agreement between annotators

    :param annotator_ids: list of annotator ids
    :param dict_annotator_x_ids: dictionary of {annotator_id: [entity/relation ids]}
    :return: agreement matrix
    """
    array_agreements = []
    for annotator1 in annotator_ids:
        temp_agreements = []
        for annotator2 in annotator_ids:
            if annotator1 == annotator2:
                temp_agreement = 1
                agreement_str = f'{len(dict_annotator_x_ids[annotator1])}/{len(dict_annotator_x_ids[annotator1])}={temp_agreement}'
            else:
                agreed_relation_count = len(
                    set(dict_annotator_x_ids[annotator1]).intersection(dict_annotator_x_ids[annotator2]))
                temp_agreement = agreed_relation_count / len(dict_annotator_x_ids[annotator1])
                agreement_str = f'{agreed_relation_count}/{len(dict_annotator_x_ids[annotator1])}={temp_agreement}'
            temp_agreements.append(temp_agreement)
            print(f'agreement {annotator1}-{annotator2}: {agreement_str}')
        array_agreements.append(temp_agreements)
    print(array_agreements)
    return array_agreements


def entity_pairwise_agreement(annotator_ids, json_path):
    """
    Calculate pairwise agreement on entity annotations

    :param annotator_ids: list of annotator ids
    :param json_path: str
    :return: agreement matrix
    """
    dict_annotator_entity_ids = dict()

    results = json.load(open(json_path))
    examples = results['examples']

    all_annotations = sum(map(lambda x: x['annotations'], examples), [])

    iaa_data_df = pd.io.json.json_normalize(all_annotations,
                                        meta=['tagged_token_id', 'example_id', 'tag', 'value', 'start', 'end'],
                                        record_path='annotated_by')
    annotations_df = iaa_data_df[['example_id', 'tagged_token_id', 'annotator_id', 'start', 'end', 'value', 'tag']]
    annotations_df['entity_id'] = annotations_df.apply(
        lambda row: f'{row["example_id"]}_{row["start"]}_{row["end"]}_{row["value"]}_{row["tag"]}', axis=1)

    for annotator_id in annotator_ids:
        dict_annotator_entity_ids[annotator_id] = (annotations_df.loc[annotations_df['annotator_id'] == annotator_id]['entity_id']).tolist()

    array_agreements = pairwise_agreement(annotator_ids, dict_annotator_entity_ids)
    return np.asmatrix(array_agreements)


def relation_pairwise_agreement(annotator_ids, list_relations):
    """
    Calculate pairwise agreement on relation annotations

    :param annotator_ids: list of annotator ids
    :param list_relations: list of BasicRelation
    :return: agreement matrix
    """
    dict_annotator_relations = {id: [] for id in annotator_ids}
    dict_annotator_relation_ids = {id: [] for id in annotator_ids}
    for relation in list_relations:
        # relation id format: {sample-id}_[parent-span]_{relation-type}_[child-span]
        # span id format: {start-index}_{end-index}_{value}_{tag}
        parent_span_id = f'{relation.parent_span["start"]}_{relation.parent_span["end"]}_{relation.parent_span["value"]}_{relation.parent_span["tag"]}'
        child_span_id = f'{relation.child_span["start"]}_{relation.child_span["end"]}_{relation.child_span["value"]}_{relation.child_span["tag"]}'
        relation_id = f'{relation.sample_id}_[{parent_span_id}]_{relation.relation_type}_[{child_span_id}]'
        dict_annotator_relations[relation.annotator_id].append(relation)
        dict_annotator_relation_ids[relation.annotator_id].append(relation_id)

    array_agreements = pairwise_agreement(annotator_ids, dict_annotator_relation_ids)
    return np.asmatrix(array_agreements)


if __name__ == '__main__':
    # annotator_ids = [1, 3]
    # annotator_ids = [3, 5, 7]  # Team A
    annotator_ids = [1, 4, 6]  # Team B
    json_path = 'data/round2/accord-r2-b_annotations.json'

    list_basic_relations = get_relations(json_path)
    list_relations = format_relations(json_path, list_basic_relations)

    m = relation_pairwise_agreement(annotator_ids, list_relations)
    print(f'matrix:\n {m}')

    # m = entity_pairwise_agreement(annotator_ids, json_path)
    # print(f'matrix:\n {m}')

    # plot_matrix(m, annotator_ids, filename='iaa/accord-test-r2.png')
