# Created by Hansi on 12/10/2023
import json
import os
import re
from os.path import isfile

import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

from experiments.analyses.eda import plot_histo, plot_bar_chart


def clean_text(text):
    cleaned_text = ' '.join(text.split())
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def regex_escape_chars(text):
    """
    Escape regex special characters
    :param text:
    :return:
    """
    text = text.replace(".", "\.")
    text = text.replace("(", "\(")
    text = text.replace(")", "\)")
    text = text.replace("/", "\/")
    text = text.replace("+", "\+")
    text = text.replace("-", "\-")
    return text


def html_to_sections(html_file, json_file):
    '''
    Extract requirement blocks from the HTML file with their nested RASE sections
    :param html_file: path to HTML file
    :param json_file: path to save output JSON
    :return:
    '''
    with open(html_file) as fp:
        soup = BeautifulSoup(fp, 'html.parser')

    # both the following divs and spans contain requirement blocks
    requirement_divs = soup.select('div[data-rasetype="RequirementSection"]')
    requirement_spans = soup.select('span[data-rasetype="RequirementSection"]')
    elements = requirement_divs + requirement_spans

    # remove spans located within divs
    inside_spans = []
    inside_divs = []
    for element in elements:
        nested_spans = element.select('span[data-rasetype="RequirementSection"]')
        nested_divs = element.select('div[data-rasetype="RequirementSection"]')
        for s in nested_spans:
            inside_spans.append(s['id'])
        for d in nested_divs:
            inside_divs.append(d['id'])

    requirement_blocks = []
    for div in requirement_divs:
        if div['id'] not in inside_divs:
            requirement_blocks.append(div)
    for span in requirement_spans:
        if span['id'] not in inside_spans:
            requirement_blocks.append(span)

    data = []

    # extract RASE sections within requirement blocks
    for i, block in enumerate(requirement_blocks):
        section_data = []

        block_text = block.get_text(separator=" ", strip=True)
        block_text = clean_text(block_text)

        # sections = []
        span_sections = block.select('span[data-rasetype*="Section"]')
        div_sections = block.select('div[data-rasetype*="Section"]')
        sections = span_sections + div_sections

        sections_df_data = []
        texts = []
        # sort section by start index
        for section in sections:
            sect_category = section['data-rasetype']
            sect_text = section.get_text(separator=" ", strip=True)
            sect_text = clean_text(sect_text)

            # handle multiple section text occurrences within a text block
            start_indices = [m.start() for m in re.finditer(regex_escape_chars(sect_text), block_text)]
            if len(start_indices) > 1:
                count = texts.count(sect_text)
                start_index = start_indices[count]
            else:
                start_index = start_indices[0]

            sections_df_data.append([start_index, len(sect_text), sect_text, sect_category])
            texts.append(sect_text)

        sections_df = pd.DataFrame(sections_df_data, columns=['s_index', 'length', 'text', 'category'])
        sections_df.sort_values(by=['s_index', 'length'], ascending=[True, False], inplace=True)

        # get nesting level
        for index, row in sections_df.iterrows():
            sect_category = row['category']
            sect_text = row['text']

            parent_id = 'B'
            level = 0
            # check for parents by iterating section data in the reverse order
            for sect in reversed(section_data):
                if sect_text in sect['text']:
                    parent_id = sect['id']
                    level = sect['level'] + 1
                    break

            section_data.append({'id': f'S{index}', 'text': sect_text, 'category': sect_category,
                                 'parent_id': parent_id, 'level': level})

        block_tokens = word_tokenize(block_text)

        block_obj = {'block_text': block_text, 'block_tokens': block_tokens, 'sections': section_data,
                     'seq_length': len(block_tokens), 'section_count': len(section_data)}
        data.append(block_obj)

    with open(json_file, "w", encoding='utf-8') as final:
        json.dump(data, final)
    return data


def html_to_sections_bulk(input_folder, output_folder):
    '''
    Extract requirement blocks from a bulk of HTML files
    :param input_folder: folder of HTML files
    :param output_folder: folder to save JSON files
    :return:
    '''
    block_count = []
    count_detailed = []
    seq_lengths = []
    section_count = []
    section_categories = []
    levels = []

    input_files = [f for f in os.listdir(input_folder) if isfile(os.path.join(input_folder, f))]
    for input_file in input_files:
        temp_section_categories = []
        temp_levels = []

        file_name = os.path.splitext(input_file)[0]
        print(f'Processing {file_name}...')
        data = html_to_sections(os.path.join(input_folder, input_file),
                                os.path.join(output_folder, f'{file_name}.json'))

        block_count.append(len(data))

        temp_seq_lengths = [d['seq_length'] for d in data]
        temp_section_counts = [d['section_count'] for d in data]

        non_empty_block_count = 0
        for d in data:
            if d['section_count'] != 0:
                non_empty_block_count += 1
            for s in d['sections']:
                temp_levels.append(s['level'])
                temp_section_categories.append(s['category'])

        count_detailed.append(
            [file_name, len(data), non_empty_block_count, sum([d['section_count'] for d in data]), temp_seq_lengths,
             temp_section_counts, [[a, temp_section_categories.count(a)] for a in set(temp_section_categories)],
             [[a, temp_levels.count(a)] for a in set(temp_levels)]])
        seq_lengths.extend(temp_seq_lengths)
        section_count.extend(temp_section_counts)
        section_categories.extend(temp_section_categories)
        levels.extend(temp_levels)

    df = pd.DataFrame(count_detailed,
                      columns=['document', 'total_blocks', 'non_empty_blocks', 'section_count', 'seq_lengths',
                               'section_counts', 'section_categories', 'levels'])
    df.to_csv(os.path.join(output_folder, 'block_counts.csv'), index=False, encoding='utf-8')

    print(f'block_count: {block_count}')
    print(f'total blocks: {sum(block_count)}')

    print('\nseq length stats:')
    print(f'seq_lengths: {seq_lengths}')
    print(f'average: {sum(seq_lengths) / len(seq_lengths)}')
    print(f'max: {max(seq_lengths)}')
    print(f'min: {min(seq_lengths)}')

    print('\nsection count (per block) stats:')
    print(f'section_counts: {section_count}')
    print(f'average: {sum(section_count) / len(section_count)}')
    print(f'max: {max(section_count)}')
    print(f'min: {min(section_count)}')
    print([[a, section_categories.count(a)] for a in set(section_categories)])

    print('\nlevel stats:')
    print(f'max: {max(levels)}')
    print(f'min: {min(levels)}')
    print([[a, levels.count(a)] for a in set(levels)])


def merge_json(input_folder, output_file):
    input_files = [f for f in os.listdir(input_folder) if isfile(os.path.join(input_folder, f))]
    all_data = []
    for input_file in input_files:
        file_name = os.path.splitext(input_file)[0]

        with open(os.path.join(input_folder, input_file)) as json_file:
            data = json.load(json_file)

        for j in data:
            j['source'] = file_name
        print(f'{file_name}: {len(data)}')
        all_data.extend(data)

    with open(output_file, "w", encoding='utf-8') as final:
        json.dump(all_data, final)


def validate_json(input_file, output_folder):
    with open(input_file) as json_file:
        data = json.load(json_file)

    validated_data = []
    data_with_sections = []
    data_without_sections = []

    for j in data:
        text = clean_text(j['block_text'])
        block_tokens = word_tokenize(text)

        for sect in j['sections']:
            sect['text'] = clean_text(sect['text'])

        block_obj = {'block_text': text, 'sections': j['sections'],
                     'seq_length': len(block_tokens), 'source': j['source']}
        validated_data.append(block_obj)
        if len(j['sections']) == 0:
            data_without_sections.append(block_obj)
        else:
            data_with_sections.append(block_obj)

    with open(os.path.join(output_folder, 'all.json'), "w", encoding='utf-8') as final:
        json.dump(validated_data, final)
    with open(os.path.join(output_folder, 'data_with_sections.json'), "w", encoding='utf-8') as final:
        json.dump(data_with_sections, final)
    with open(os.path.join(output_folder, 'data_without_sections.json'), "w", encoding='utf-8') as final:
        json.dump(data_without_sections, final)


def get_stats(json_file, plot_folder):
    with open(json_file) as j_file:
        data = json.load(j_file)

    seq_lengths = []
    levels = []
    section_categories = []

    for j in data:
        seq_lengths.append(j['seq_length'])
        for sect in j['sections']:
            levels.append(sect['level'])
            section_categories.append(sect['category'])

    print('\nseq length stats:')
    plot_histo(seq_lengths, plot_path=os.path.join(plot_folder, 'seq_length_histo.png'))

    print('\nlevel stats:')
    print(f'max: {max(levels)}')
    print(f'min: {min(levels)}')
    print([[a, levels.count(a)] for a in set(levels)])

    print('\nsection category stats:')
    print([[a, section_categories.count(a)] for a in set(section_categories)])
    plot_bar_chart(section_categories, plot_path=os.path.join(plot_folder, 'categories.png'))


if __name__ == '__main__':
    html_file = '../data/input/BB100-Section4.html'
    json_file = '../data/test.json'
    # html_to_sections(html_file, json_file)

    input_folder = '../data/input'
    output_folder = '../data/output2'
    # html_to_sections_bulk(input_folder, output_folder)

    input_folder = '../../data/rase/validated-text-blocks'
    output_file = '../../data/rase/validated-text-blocks.json'
    # merge_json(input_folder, output_file)

    # validate_json('../../data/rase/validated-text-blocks.json', '../../data/rase/final_data')

    get_stats('../../data/rase/final_data/data_with_sections.json', '../../data/rase/final_data/plots')
