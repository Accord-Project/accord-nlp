# Created by Hansi at 26/04/2023

import json


def clean_json(input_json_path, example_ids, output_json_path):
    """
    Remove all annotations correspond to the given example ids

    :param input_json_path: str
    :param example_ids: list of str
    :param output_json_path: str
    :return:
    """
    results = json.load(open(input_json_path))
    examples = results['examples']
    relations = results['relations']

    example_indices = []
    relation_indices = []

    tagged_token_ids = []
    for example in examples:
        if example['example_id'] in example_ids:
            example_indices.append(examples.index(example))

            example_tagged_token_ids = [annotation['tagged_token_id'] for annotation in example['annotations']]
            tagged_token_ids.extend(example_tagged_token_ids)

    for relation in relations:
        if relation['tagged_token_id'] in tagged_token_ids:
            relation_indices.append(relations.index(relation))

    updated_examples = [examples[i] for i in range(len(examples)) if i not in example_indices]
    updated_relations = [relations[i] for i in range(len(relations)) if i not in relation_indices]

    results['examples'] = updated_examples
    results['relations'] = updated_relations

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f)


if __name__ == '__main__':
    input_json_path = 'data/test-round2/accord-testround2-v2_annotations.json'
    example_ids = ['48361556-520b-4a1f-9338-11d5292a6360']
    output_json_path = 'data/test-round2/accord-testround2-v2_annotations_cleaned.json'
    clean_json(input_json_path, example_ids, output_json_path)
