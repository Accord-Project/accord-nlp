# Created by Hansi at 28/06/2023
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(file_path, output_folder, test_size=0.2, class_column=None):
    df = pd.read_csv(file_path, encoding='utf-8')
    stratify = None
    if class_column is not None:
        stratify = df[class_column].tolist()

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=157, stratify=stratify)
    print(f'train shape: {train_df.shape}')
    print(f'test shape: {test_df.shape}')
    train_df.to_csv(os.path.join(output_folder, 'train.csv'), encoding='utf-8', index=False)
    test_df.to_csv(os.path.join(output_folder, 'test.csv'), encoding='utf-8', index=False)


def concat_data(data_paths, counts, output_path):
    df_list = []
    for i in range(len(data_paths)):
        df = pd.read_csv(data_paths[i], encoding='utf-8')
        if counts[i] != -1:
            df = df.sample(n=counts[i], replace=False, random_state=157)
        df_list.append(df)
    concat_df = pd.concat(df_list)
    concat_df = concat_df.sample(frac=1).reset_index(drop=True)
    concat_df.to_csv(output_path, encoding='utf-8', index=False)


def format_ner_data(df):
    """
    Convert dataframe to NER model's input format
    :param df:
    :return:
    """
    token_lst = []
    for index, row in df.iterrows():
        sentence_id = row["example_id"]
        tokens = row["processed_content"].split()
        labels = row["label"].split()
        for token, label in zip(tokens, labels):
            token_lst.append([sentence_id, token, label])

    token_df = pd.DataFrame(token_lst, columns=["sentence_id", "words", "labels"])
    return token_df


if __name__ == '__main__':
    # file_path = '../data/ner/processed-entities.csv'
    # output_folder = '../data/ner'
    # split_data(file_path, output_folder)

    # data_paths = ['../data/re/relations.csv', '../data/re/none_relations.csv']
    # counts = [-1, 1000]
    # output_path = '../data/re/all.csv'
    # concat_data(data_paths, counts, output_path)

    file_path = '../data/re/all.csv'
    output_folder = '../data/re'
    split_data(file_path, output_folder, class_column='relation_type')
