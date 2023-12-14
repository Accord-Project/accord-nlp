# Created by Hansi at 28/06/2023
import argparse
import os
import shutil

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from accord_nlp.text_classification.ner.ner_model import NERModel
from experiments.ner.evaluation import print_eval_ner, cls_report, strict_cls_report
from experiments.ner.ner_config import ner_args, SEED
from experiments.utils import format_ner_data

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
ner_args['wandb_project'] = 'ner'
wandb_run_name = None
if arguments.wandb_api_key is not None:
    os.environ['WANDB_API_KEY'] = arguments.wandb_api_key

if arguments.wandb_run_name is not None:
    ner_args['wandb_kwargs'] = {'name': arguments.wandb_run_name}
else:
    ner_args['wandb_kwargs'] = {
        'name': f"{MODEL_NAME.split('/')[-1]}_{ner_args['learning_rate']}_{ner_args['num_train_epochs']}"}

train_file_path = "data/ner/train.csv"
test_file_path = "data/ner/test.csv"
train_df = pd.read_csv(train_file_path, encoding='utf-8')
test_df = pd.read_csv(test_file_path, encoding='utf-8')

train, eval = train_test_split(train_df, test_size=0.1, random_state=SEED)
print(f'train size: {train.shape}')
print(f'eval size: {eval.shape}')
train_token_df = format_ner_data(train)
eval_token_df = format_ner_data(eval)

test_token_df = format_ner_data(test_df)

# tags = train_token_df['labels'].unique().tolist()
model = NERModel(MODEL_TYPE, MODEL_NAME, labels=ner_args['labels_list'], use_cuda=torch.cuda.is_available(),
                 cuda_device=cuda_device, args=ner_args)

model.train_model(train_token_df, eval_df=eval_token_df, cls_report=cls_report, strict_cls_report=strict_cls_report)

model = NERModel(MODEL_TYPE, ner_args['best_model_dir'], labels=ner_args['labels_list'],
                 use_cuda=torch.cuda.is_available(), cuda_device=cuda_device, args=ner_args)
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
                os.path.join(ner_args['best_model_dir'], f"training_progress_scores.csv"))
