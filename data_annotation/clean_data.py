# Created by Hansi at 02/05/2023
import re

import numpy as np
import pandas as pd


def identify_duplicates(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    all_df = pd.merge(df1, df2, on=['Cleaned_Text'], how='left', indicator='exists')
    all_df['exists'] = np.where(all_df.exists == 'both', 'R3-H', 0)

    all_df.to_csv(output_path, index=False, encoding='utf-8')


def identify_own_duplicates(file_path, output_path):
    df = pd.read_csv(file_path)

    df['Duplicate'] = df.duplicated(subset=['Cleaned_Text'])

    # df.drop_duplicates(subset='Cleaned_Text', inplace=True)
    df.to_csv(output_path, index=False, encoding='utf-8')


def clean_text_df(file_path, cleaned_file_path):
    df = pd.read_csv(file_path)

    # compile regex to match multiple space occurrences in the text
    pattern = '\s\s+'
    pat = re.compile(pattern)

    df['Cleaned_Text'] = df.apply(lambda row: row['Text'].strip(), axis=1)
    df['Cleaned_Text'] = df.apply(lambda row: pat.sub(' ', row['Cleaned_Text']), axis=1)

    df.to_csv(cleaned_file_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    file1_path = 'data/UK-All-self-contained-Sentences.csv'
    file2_path = 'data/round3/Data_Round3_TeamH.csv'
    output_path = 'data/output.csv'
    identify_duplicates(file1_path, file2_path, output_path)

    # file_path = 'data/UK-All-self-contained-Sentences.csv'
    # cleaned_file_path = 'data/cleaned.csv'
    # clean_text_df(file_path, cleaned_file_path)

    # file_path='data/UK-All-self-contained-Sentences.csv'
    # output_path= 'data/without_dups.csv'
    # drop_duplicates(file_path, output_path)
