# Created by Hansi at 22/08/2023
import argparse
import os
import shutil

import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

from experiments.relation_extraction.evaluation import macro_f1, macro_recall, macro_precision, cls_report, \
    print_eval_results
from experiments.relation_extraction.re_config import re_args, SEED
from accord_nlp.text_classification.relation_extraction.re_model import REModel

parser = argparse.ArgumentParser(description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--k_folds', required=False, help='k folds', default=5)
# set if the wandb login cannot be allowed during the run
parser.add_argument('--wandb_api_key', required=False, help='wandb api key', default=None)
parser.add_argument('--wandb_run_name', required=False, help='wandb run name', default=None)
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)
k_folds = int(arguments.k_folds)
re_args['wandb_project'] = 'relation-extraction-cv'
if arguments.wandb_api_key is not None:
    os.environ['WANDB_API_KEY'] = arguments.wandb_api_key

folds = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

data_file_path = "data/re/all.csv"
data_df = pd.read_csv(data_file_path, encoding='utf-8')
data_df = data_df.rename(columns={'tagged_sentence': 'text', 'relation_type': 'labels'})
data_df = data_df[['example_id', 'text', 'labels']]
print(f'data size: {data_df.shape}')

splits = folds.split(data_df)
fold_i = 0
base_best_model_dir = re_args['best_model_dir']
os.makedirs(base_best_model_dir, exist_ok=True)

all_predictions = []
all_actual_labels = []

for train, test in splits:
    print(f'fold {fold_i}')
    if arguments.wandb_run_name is not None:
        re_args['wandb_kwargs'] = {'group': arguments.wandb_run_name, 'job_type': str(fold_i)}
    else:
        re_args['wandb_kwargs'] = {'group': f"{MODEL_NAME.split('/')[-1]}_{re_args['learning_rate']}_{re_args['num_train_epochs']}", 'job_type': str(fold_i)}

    print('train: %s, test: %s' % (data_df.iloc[train].shape, data_df.iloc[test].shape))
    train_df = data_df.iloc[train]
    test_df = data_df.iloc[test]

    train, eval = train_test_split(train_df, test_size=0.1, random_state=SEED)
    print(f'train size: {train.shape}')
    print(f'eval size: {eval.shape}')

    re_args['best_model_dir'] = os.path.join(base_best_model_dir, f'fold_{fold_i}')

    re_args['labels_list'] = train_df['labels'].unique().tolist()

    model = REModel(MODEL_TYPE, MODEL_NAME, use_cuda=torch.cuda.is_available(), args=re_args)
    model.train_model(train, eval_df=eval, macro_f1=macro_f1, macro_r=macro_recall, macro_p=macro_precision,
                      cls_report=cls_report)

    model = REModel(MODEL_TYPE, re_args['best_model_dir'], use_cuda=torch.cuda.is_available(), args=re_args)
    predictions, raw_outputs = model.predict(test_df["text"].tolist())
    test_df['predictions'] = predictions

    print_eval_results(test_df['labels'].tolist(), predictions,
                       eval_file_path=os.path.join(re_args['best_model_dir'], 'test_eval.txt'))
    test_df.to_csv(os.path.join(re_args['best_model_dir'], 'predictions.csv'), encoding='utf-8', index=False)

    shutil.copyfile(os.path.join(re_args['output_dir'], "training_progress_scores.csv"),
                    os.path.join(base_best_model_dir, f"training_progress_scores_{fold_i}.csv"))

    all_predictions.extend(predictions)
    all_actual_labels.extend(test_df['labels'].tolist())
    fold_i = fold_i + 1

# evaluation of all folds
print_eval_results(all_actual_labels, all_predictions, eval_file_path=os.path.join(base_best_model_dir, 'full_eval.txt'))


