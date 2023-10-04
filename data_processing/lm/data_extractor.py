# Created by Hansi on 22/09/2023

import os
# nltk.download('punkt')
import re

import nltk
import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg")

tag_line_break = '[LINE-BREAK]'

symbols_regex = "^[\*_—–]"  # symbols
pattern_symbols = re.compile(symbols_regex)

number_regex = "^\(?\d+(\.\d*)?[\s\)\.]"  # number only
pattern_numbers = re.compile(number_regex)  # compile pattern into regular expression object

roman_numbers_regex = "^\(?i{0,3}(x|v)?i{0,3}(\)|\.)|^\(?I{0,3}(X|V)?I{0,3}(\)|\.)"  # roman numbers (i - Xiii)
pattern_roman_numbers = re.compile(roman_numbers_regex)

letters_regex = "^\(?\d+[a-zA-Z]\d*[\s\.\)]|^\(?\d*[a-zA-Z]\d+[\s\.\)]|^\(?[a-zA-Z][\.\)]"  # letter-based pointers (A1, 1A2, a., etc.)
pattern_letters = re.compile(letters_regex)

end_regex = "[\.\?:;,]$|(and|or)$"
pattern_end = re.compile(end_regex)  # compile pattern into regular expression object

word_regex = "[a-zA-Z]+"
pattern_word = re.compile(word_regex)  # compile pattern into regular expression object


def split_sentences_nltk(text):
    return nltk.sent_tokenize(text)


def split_sentences_spacy(text):
    nlp.disable_pipe("parser")
    nlp.enable_pipe("senter")
    doc = nlp(text)
    return list(doc.sents)


def remove_pointers(text):
    sub_pointer_patterns = [pattern_symbols, pattern_numbers, pattern_roman_numbers]

    has_pointers = True
    while has_pointers:  # remove pointers at different levels (e.g., G1.  (1))
        text = pattern_symbols.sub('', text)
        text = pattern_letters.sub('', text)  # remove pointers with letters
        text = pattern_numbers.sub('', text)  # remove pointers with numbers
        text = pattern_roman_numbers.sub('', text)  # remove pointers with roman numbers
        text = text.strip()

        has_pointers = False
        for pattern in sub_pointer_patterns:
            pattern_output = pattern.match(text)
            if pattern_output is not None and pattern_output.span() != (0, 0):
                has_pointers = True
                break

    return text


def extract_sentences(txt_file, output_folder, domain):
    # spacy pipeline for sentence splitting
    nlp.disable_pipe("parser")
    nlp.enable_pipe("senter")

    with open(txt_file, encoding='utf-8') as fr:
        text = fr.read()

    # merge text of each paragraph
    merged_text = re.sub('\n\n', tag_line_break, text)
    merged_text = re.sub('\n', ' ', merged_text)
    splits = merged_text.split(tag_line_break)

    cleaned_splits = []
    all_text = []
    sentences = []
    non_sentences = []
    i = 0
    for text in tqdm(splits):
        # clean text
        text = ' '.join(text.split())  # remove additional spaces
        text = remove_pointers(text)  # remove pointers
        text = text.strip()  # remove additional spaces at the beginning and end
        cleaned_splits.append(text)

        # split into sentences
        text_list = split_sentences_spacy(text)

        for text_unit in text_list:
            id = f'{i}_UK_{domain}'
            all_text.append([id, text_unit])
            if len(pattern_end.findall(str(text_unit))) == 0 or len(pattern_word.findall(str(text_unit))) < 3:
                non_sentences.append([id, text_unit])
            else:
                sentences.append([id, text_unit])
            i += 1

    df_all = pd.DataFrame(all_text, columns=['id', 'text'])
    df_sent = pd.DataFrame(sentences, columns=['id', 'text'])
    df_non_sent = pd.DataFrame(non_sentences, columns=['id', 'text'])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df_all.to_csv(os.path.join(output_folder, 'all.csv'), encoding='utf-8', index=False)
    df_sent.to_csv(os.path.join(output_folder, 'sentences.csv'), encoding='utf-8', index=False)
    df_non_sent.to_csv(os.path.join(output_folder, 'non_sentences.csv'), encoding='utf-8', index=False)


if __name__ == '__main__':
    chapter = 'Approved_Document_A'
    domain = 'DocA_Structure'

    txt_file = os.path.join('../../data/sentences/txt', f'Text-{chapter}-Content.txt')
    output_folder = '../../data/lm/DocA_Structure'

    extract_sentences(txt_file, output_folder, domain)
