# Created by Hansi at 17/07/2023
import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from experiments.ner.ner_config import SEED, ner_args
from experiments.ner.ner_experiments import format_data
from experiments.utils import print_eval_ner
from text_classification.ner.ner_model import NERModel

parser = argparse.ArgumentParser(description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--k_folds', required=False, help='k folds', default=2)
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)
k_folds = int(arguments.k_folds)

folds = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

data_file_path = "../../data/ner/all.csv"
data_df = pd.read_csv(data_file_path, encoding='utf-8')
data_df = data_df.head(100)
print(f'data size: {data_df.shape}')

splits = folds.split(data_df)
fold_i = 0
base_best_model_dir = ner_args['best_model_dir']
os.makedirs(base_best_model_dir, exist_ok=True)

for train, test in splits:
    print(f'fold {fold_i}')
    ner_args['wandb_kwargs'] = {'group': MODEL_NAME.split('/')[-1], 'job_type': fold_i}

    print('train: %s, test: %s' % (data_df.iloc[train].shape, data_df.iloc[test].shape))
    train_df = data_df.iloc[train]
    test_df = data_df.iloc[test]

    train, eval = train_test_split(train_df, test_size=0.1, random_state=SEED)
    print(f'train size: {train.shape}')
    print(f'eval size: {eval.shape}')
    train_token_df = format_data(train)
    eval_token_df = format_data(eval)

    test_token_df = format_data(test_df)

    tags = train_token_df['labels'].unique().tolist()
    ner_args['best_model_dir'] = os.path.join(base_best_model_dir, f'fold_{fold_i}')

    model = NERModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=ner_args)
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
    print_eval_ner(actual_labels, converted_predictions,
                   eval_file_path=os.path.join(ner_args['best_model_dir'], 'test_eval.txt'))

    flat_predictions = [j for sub in converted_predictions for j in sub]
    test_token_df["predictions"] = flat_predictions
    test_token_df.to_csv(os.path.join(ner_args['best_model_dir'], 'predictions.csv'), encoding='utf-8', index=False)

    shutil.copyfile(os.path.join(ner_args['output_dir'], "training_progress_scores.csv"),
                    os.path.join(ner_args['best_model_dir'], f"training_progress_scores_{fold_i}.csv"))
    fold_i = fold_i + 1

