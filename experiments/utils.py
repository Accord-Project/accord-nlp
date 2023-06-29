# Created by Hansi at 28/06/2023
import os

import pandas as pd
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from sklearn.model_selection import train_test_split


def split_data(file_path, output_folder, test_size=0.2):
    df = pd.read_csv(file_path, encoding='utf-8')
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=157)
    print(f'train shape: {train_df.shape}')
    print(f'test shape: {test_df.shape}')
    train_df.to_csv(os.path.join(output_folder, 'train.csv'), encoding='utf-8', index=False)
    test_df.to_csv(os.path.join(output_folder, 'test.csv'), encoding='utf-8', index=False)


def print_eval_ner(actuals, preds, eval_file_path=None):
    f = None
    if eval_file_path is not None:
        f = open(eval_file_path, "w")

    cls_report = classification_report(actuals, preds)
    print("Default classification report:\n")
    print(cls_report)
    if eval_file_path is not None:
        f.write("Default classification report:\n")
        f.write("{}\n".format(cls_report))

    cls_report_strict = classification_report(actuals, preds, mode="strict", scheme=IOB2)
    print("Strict classification report:\n")
    print(cls_report_strict)

    if eval_file_path is not None:
        f.write("Strict classification report:\n")
        f.write("{}\n".format(cls_report_strict))

    result = {
        "precision(macro)": precision_score(actuals, preds, average="macro"),
        "recall(macro)": recall_score(actuals, preds, average="macro"),
        "f1_score(macro)": f1_score(actuals, preds, average="macro"),
        "precision_strict(macro)": precision_score(actuals, preds, average="macro", mode="strict", scheme=IOB2),
        "recall_strict(macro)": recall_score(actuals, preds, average="macro", mode="strict", scheme=IOB2),
        "f1_score_strict(macro)": f1_score(actuals, preds, average="macro", mode="strict", scheme=IOB2)
    }

    for key in result.keys():
        print(f'{key} = {result[key]}')
        if eval_file_path is not None:
            f.write(f"{key} = {str(result[key])}\n")

    if eval_file_path is not None:
        f.close()


if __name__ == '__main__':
    file_path = '../data/ner/processed-entities.csv'
    output_folder = '../data/ner'
    split_data(file_path, output_folder)


