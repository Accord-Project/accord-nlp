# Created by Hansi at 27/06/2023
# import nltk
# nltk.download('punkt')

import re

import pandas as pd
from nltk.tokenize import word_tokenize


def validate_indices(input_path, output_path):
    df = pd.read_excel(input_path, sheet_name='Entities')
    validation = []

    for index, row in df.iterrows():
        entity = (row['content'])[int(row['start']): int(row['end'])]
        v = 1 if entity == str(row['value']) else 0
        validation.append(v)

    df['validation'] = validation
    df.to_csv(output_path, encoding='utf-8', index=False)


def regex_escape_chars(text):
    """
    Escape regex special characters
    :param text:
    :return:
    """
    if text == '.': text = "\."
    text = text.replace("(", "\(")
    text = text.replace(")", "\)")
    return text


def process_sentence(sentence):
    """
    Convert a sentence into the format of {token1-s-e: token1, token2-s-e: token2, ...}.
    Tokeniser - NLTK word_tokenize

    :param sentence:
    :return:
        error = 0 or 1
        dict_tokens
    """
    error = 0
    dict_tokens = {}
    final_tokens = []
    final_indices = []

    tokens = re.split(r'\s+', sentence)  # tokenise using spaces
    if tokens[0] == '': del tokens[0]
    if tokens[len(tokens) - 1] == '': del tokens[len(tokens) - 1]

    token_indices = [(ele.start(), ele.end()) for ele in re.finditer(r'\S+', sentence)]
    if (len(tokens) == len(token_indices)):
        for i in range(len(tokens)):
            sub_tokens = word_tokenize(tokens[i])  # tokenise using NLTK
            if len(sub_tokens) > 1:
                token_start = token_indices[i][0]
                for sub_token in sub_tokens:
                    final_tokens.append(sub_token)

                    escaped_sub_token = regex_escape_chars(sub_token)
                    final_indices.append([(token_start + ele.start(), token_start + ele.end()) for ele in
                                          re.finditer(escaped_sub_token, tokens[i])][0])
            else:
                final_tokens.append(tokens[i])
                final_indices.append(token_indices[i])

        for i in range(len(final_tokens)):
            dict_tokens[f'{final_tokens[i]}-{final_indices[i][0]}-{final_indices[i][1]}'] = final_tokens[i]
    else:
        print(f'Error: Mismatching tokens and indices were found for Sentence {id}.')
        error = 1

    return error, dict_tokens


def process_entities(sentence, filtered_df):
    """
    Convert the entities of a sentence into the format of {token_x-s-e: [token_x, label], token_y-s-e: [token_y, label], ...}.
    Tokeniser - NLTK word_tokenize

    :param sentence:
    :param filtered_df:
    :return:
        dict_entities
    """
    dict_entities = {}
    for index, row in filtered_df.iterrows():
        entity_text = sentence[int(row['start']): int(row['end'])]
        label = row['tag']

        entity_tokens = word_tokenize(entity_text)
        i = 0
        for entity_token in entity_tokens:
            escaped_sub_token = regex_escape_chars(entity_token)
            index_pair = [(int(row['start']) + ele.start(), int(row['start']) + ele.end()) for ele in
                          re.finditer(escaped_sub_token, entity_text)][0]
            if i == 0:
                dict_entities[f'{entity_token}-{index_pair[0]}-{index_pair[1]}'] = [entity_token, f'B-{label}']
            else:
                dict_entities[f'{entity_token}-{index_pair[0]}-{index_pair[1]}'] = [entity_token, f'I-{label}']
            i = i + 1

    return dict_entities


def format_iob(input_path, output_path):
    """
    Format strings to IOB format
    :param input_path: .csv (final-entities)
        compulsory columns: example_id, content, metadata, start, end, value, tag
    :param output_path: .csv
    :return:
    """
    df = pd.read_csv(input_path, encoding='utf-8')

    ids = df['example_id'].unique().tolist()

    output_df = pd.DataFrame(columns=['example_id', 'content', 'label', 'metadata'])

    for n in range(len(ids)):
        id = ids[n]
        filtered_df = df.loc[df['example_id'] == id]
        row0 = filtered_df.iloc[0]
        sentence = row0['content']
        metadata = row0['metadata']
        error, dict_tokens = process_sentence(sentence)

        if error != 1:
            dict_entities = process_entities(sentence, filtered_df)

            processed_sentence = ' '.join(dict_tokens.values())
            iob_string = ''
            i = 0
            for k, v in dict_tokens.items():
                token_label = 'O'
                if k in dict_entities:
                    token_label = dict_entities[k][1]

                if i == 0:
                    iob_string = token_label
                else:
                    iob_string = f'{iob_string} {token_label}'
                i = i + 1

            output_df.loc[n] = [id, processed_sentence, iob_string, metadata]

        output_df.to_csv(output_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    input_path = '../../data/final/Final-Data.xlsx'
    output_path = '../../data/final/validated-entities.csv'
    # validate_indices(input_path, output_path)

    input_path = '../../data/final/final-entities.csv'
    output_path = '../../data/final/processed-entities2.csv'
    format_iob(input_path, output_path)
