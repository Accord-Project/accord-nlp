# Created by Hansi at 28/06/2023
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.ner.ner_config import ner_args, SEED
from experiments.utils import print_eval_ner
from text_classification.ner.ner_model import NERModel


def format_data(df):
    token_lst = []
    for index, row in df.iterrows():
        sentence_id = row["example_id"]
        tokens = row["processed_content"].split()
        labels = row["label"].split()
        for token, label in zip(tokens, labels):
            token_lst.append([sentence_id, token, label])

    token_df = pd.DataFrame(token_lst, columns=["sentence_id", "words", "labels"])
    return token_df


parser = argparse.ArgumentParser(description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

train_file_path = "data/ner/train.csv"
test_file_path = "data/ner/test.csv"
train_df = pd.read_csv(train_file_path, encoding='utf-8')
test_df = pd.read_csv(test_file_path, encoding='utf-8')

train, eval = train_test_split(train_df, test_size=0.1, random_state=SEED)
print(f'train size: {train.shape}')
print(f'eval size: {eval.shape}')
train_token_df = format_data(train)
eval_token_df = format_data(eval)

test_token_df = format_data(test_df)

# train_token_df = []
# # train_sentence_id = 0
# for index, row in train_df.iterrows():
#     train_sentence_id = row["example_id"]
#     tokens = row["processed_content"].split()
#     labels = row["label"].split()
#     for token, label in zip(tokens, labels):
#         train_token_df.append([train_sentence_id, token, label])
#     # train_sentence_id = train_sentence_id + 1
#
# train_data = pd.DataFrame(train_token_df, columns=["sentence_id", "words", "labels"])
#
# test_token_df = []
# # test_sentence_id = 0
# for index, row in test_df.iterrows():
#     test_sentence_id = row["example_id"]
#     tokens = row["processed_content"].split()
#     labels = row["label"].split()
#     for token, label in zip(tokens, labels):
#         test_token_df.append([test_sentence_id, token, label])
#     # test_sentence_id = test_sentence_id + 1
#
# test_data = pd.DataFrame(test_token_df, columns=["sentence_id", "words", "labels"])

tags = train_token_df['labels'].unique().tolist()
model = NERModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=ner_args)

# train_df, eval_df = train_test_split(train_data, test_size=0.1, random_state=SEED)
model.train_model(train_token_df, eval_df=eval_token_df)

model = NERModel(MODEL_TYPE, ner_args['best_model_dir'], labels=tags, args=ner_args)
predictions, raw_outputs = model.predict(test_df["content"].tolist())
final_predictions = []
for prediction in predictions:
    raw_prediction = []
    for word_prediction in prediction:
        for key, value in word_prediction.items():
            raw_prediction.append(value)
    final_predictions.append(raw_prediction)

sentences = test_df["processed_content"].tolist()
converted_predictions = []
for final_prediction, sentence in zip(final_predictions, sentences):
    final_prediction += (len(sentence.split()) - len(final_prediction)) * ["O"]
    converted_predictions.append(final_prediction)

actuals = test_df["label"].tolist()
actual_labels = [sub.split() for sub in actuals]
print_eval_ner(actual_labels, converted_predictions, eval_file_path=os.path.join(ner_args['output_dir'], 'eval.txt'))

flat_predictions = [j for sub in converted_predictions for j in sub]
test_token_df["predictions"] = flat_predictions
test_token_df.to_csv(os.path.join(ner_args['output_dir'], 'predictions.csv'), encoding='utf-8', index=False)
