# Created by Hansi at 21/07/2023
import argparse
import os
import shutil

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from experiments.relation_extraction.evaluation import macro_f1, macro_recall, macro_precision, print_eval_results, \
    cls_report
from experiments.relation_extraction.re_config import re_args, SEED
from text_classification.relation_extraction.re_model import REModel


# def test():

parser = argparse.ArgumentParser(description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
# set if the wandb login cannot be allowed during the run
parser.add_argument('--wandb_api_key', required=False, help='wandb api key', default=None)
parser.add_argument('--wandb_run_name', required=False, help='wandb run name', default=None)
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)
re_args['wandb_project'] = 'relation-extraction'

# MODEL_TYPE = 'bert'
# MODEL_NAME = 'bert-base-cased'

wandb_run_name = None
if arguments.wandb_api_key is not None:
    os.environ['WANDB_API_KEY'] = arguments.wandb_api_key

if arguments.wandb_run_name is not None:
    re_args['wandb_kwargs'] = {'name': arguments.wandb_run_name}
else:
    re_args['wandb_kwargs'] = {'name': f"{MODEL_NAME.split('/')[-1]}_{re_args['learning_rate']}_{re_args['num_train_epochs']}"}

train_file_path = "data/re/train.csv"
test_file_path = "data/re/test.csv"
train_df = pd.read_csv(train_file_path, encoding='utf-8')
train_df = train_df.rename(columns={'tagged_sentence': 'text', 'relation_type': 'labels'})
train_df = train_df[['example_id', 'text', 'labels']]

train, eval = train_test_split(train_df, test_size=0.1, random_state=SEED, stratify=train_df['labels'].tolist())
print(f'train size: {train.shape}')
print(f'eval size: {eval.shape}')

test_df = pd.read_csv(test_file_path, encoding='utf-8')
test_df = test_df.rename(columns={'tagged_sentence': 'text', 'relation_type': 'labels'})
test_df = test_df[['example_id', 'text', 'labels']]

re_args['labels_list'] = train_df['labels'].unique().tolist()
model = REModel(MODEL_TYPE, MODEL_NAME, use_cuda=torch.cuda.is_available(), args=re_args)
model.train_model(train, eval_df=eval, macro_f1=macro_f1, macro_r=macro_recall, macro_p=macro_precision, cls_report=cls_report)

model = REModel(MODEL_TYPE, re_args['best_model_dir'], use_cuda=torch.cuda.is_available(), args=re_args)

predictions, raw_outputs = model.predict(test_df["text"].tolist())
test_df['predictions'] = predictions

print_eval_results(test_df['labels'].tolist(), predictions, eval_file_path=os.path.join(re_args['best_model_dir'], 'test_eval.txt'))
test_df.to_csv(os.path.join(re_args['best_model_dir'], 'predictions.csv'), encoding='utf-8', index=False)

shutil.copyfile(os.path.join(re_args['output_dir'], "training_progress_scores.csv"),
                os.path.join(re_args['best_model_dir'], f"training_progress_scores.csv"))


# if __name__ == '__main__':
#     test()
