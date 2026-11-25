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
    return re.escape(text)


def find_token_spans(token, text, offset=0):
    """
    Find the character spans of a token in text.
    NLTK's tokeniser rewrites double quotes as `` and '', which do not occur in the original text.
    """
    if token in ('``', "''"):
        pattern = '"|``|' + "''"
    else:
        pattern = regex_escape_chars(token)
    return [(offset + ele.start(), offset + ele.end()) for ele in re.finditer(pattern, text)]


def select_span(index_pairs, last_end):
    """
    Select the span closest to (and not before) the end of the previously processed token
    """
    if not index_pairs:
        raise ValueError('a token could not be located in the text')
    if len(index_pairs) == 1:
        return index_pairs[0]
    candidates = [(pair[0] - last_end, i) for i, pair in enumerate(index_pairs) if pair[0] - last_end >= 0]
    if not candidates:
        return index_pairs[-1]
    return index_pairs[min(candidates)[1]]


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
                last_index_pair = (0, 0)
                for sub_token in sub_tokens:
                    final_tokens.append(sub_token)

                    index_pairs = find_token_spans(sub_token, tokens[i], token_start)
                    index_pair = select_span(index_pairs, last_index_pair[1])
                    final_indices.append(index_pair)
                    last_index_pair = index_pair
            else:
                final_tokens.append(tokens[i])
                final_indices.append(token_indices[i])

        for i in range(len(final_tokens)):
            dict_tokens[f'{final_tokens[i]}-{final_indices[i][0]}-{final_indices[i][1]}'] = final_tokens[i]
    else:
        print(f'Error: Mismatching tokens and indices were found for sentence "{sentence[:60]}".')
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
        last_index_pair = (0, 0)
        for entity_token in entity_tokens:
            index_pairs = find_token_spans(entity_token, entity_text, int(row['start']))
            index_pair = select_span(index_pairs, last_index_pair[1])

            if i == 0:
                dict_entities[f'{entity_token}-{index_pair[0]}-{index_pair[1]}'] = [entity_token, f'B-{label}']
            else:
                dict_entities[f'{entity_token}-{index_pair[0]}-{index_pair[1]}'] = [entity_token, f'I-{label}']
            i = i + 1
            last_index_pair = index_pair

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

    output_df = pd.DataFrame(columns=['example_id', 'content', 'processed_content', 'label', 'metadata'])

    for n in range(len(ids)):
        id = ids[n]
        filtered_df = df.loc[df['example_id'] == id]
        row0 = filtered_df.iloc[0]
        sentence = row0['content']
        metadata = row0['metadata']
        try:
            error, dict_tokens = process_sentence(sentence)
            dict_entities = process_entities(sentence, filtered_df) if error != 1 else None
        except ValueError as e:
            print(f'Error: {e} in sentence {id}.')
            continue

        if error != 1:

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

            if len(processed_sentence.split(' ')) == len(iob_string.split(' ')):
                output_df.loc[n] = [id, sentence, processed_sentence, iob_string, metadata]
            else:
                print(f'Error: Mismatching tokens and labels were found for Sentence {id}.')

    output_df.to_csv(output_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    input_path = '../../data/Final-Data.xlsx'
    output_path = '../../data/ner/validated-entities.csv'
    # validate_indices(input_path, output_path)

    input_path = '../../data/ner/validated-entities.csv'
    output_path = '../../data/ner/processed-entities.csv'
    # format_iob(input_path, output_path)


