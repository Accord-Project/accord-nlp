# Created by Hansi on 17/07/2023

# NER cross-validation experiment
import argparse
import os
import shutil

from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import KFold, train_test_split

from accord_nlp.text_classification.ner.ner_model import NERModel
from experiments.ner.evaluation import print_eval_ner
from experiments.ner.ner_config import SEED, ner_args
from experiments.utils import format_ner_data

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
ner_args['wandb_project'] = 'ner-cv'
if arguments.wandb_api_key is not None:
    os.environ['WANDB_API_KEY'] = arguments.wandb_api_key

folds = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

train_df = Dataset.to_pandas(load_dataset('ACCORD-NLP/CODE-ACCORD-Entities', split='train'))
test_df = Dataset.to_pandas(load_dataset('ACCORD-NLP/CODE-ACCORD-Entities', split='test'))
data_df = train_df.append(test_df, ignore_index=True)
print(f'data size: {data_df.shape}')

splits = folds.split(data_df)
fold_i = 0
base_best_model_dir = ner_args['best_model_dir']
os.makedirs(base_best_model_dir, exist_ok=True)

all_converted_predictions = []
all_actual_labels = []

for train, test in splits:
    print(f'fold {fold_i}')
    if arguments.wandb_run_name is not None:
        ner_args['wandb_kwargs'] = {'group': arguments.wandb_run_name, 'job_type': str(fold_i)}
    else:
        ner_args['wandb_kwargs'] = {
            'group': f"{MODEL_NAME.split('/')[-1]}_{ner_args['learning_rate']}_{ner_args['num_train_epochs']}",
            'job_type': str(fold_i)}

    print('train: %s, test: %s' % (data_df.iloc[train].shape, data_df.iloc[test].shape))
    train_df = data_df.iloc[train]
    test_df = data_df.iloc[test]

    train, eval = train_test_split(train_df, test_size=0.1, random_state=SEED)
    print(f'train size: {train.shape}')
    print(f'eval size: {eval.shape}')
    train_token_df = format_ner_data(train)
    eval_token_df = format_ner_data(eval)

    test_token_df = format_ner_data(test_df)

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
                    os.path.join(base_best_model_dir, f"training_progress_scores_{fold_i}.csv"))

    all_converted_predictions.extend(converted_predictions)
    all_actual_labels.extend(actual_labels)
    fold_i = fold_i + 1

# evaluation of all folds
print_eval_ner(all_actual_labels, all_converted_predictions,
               eval_file_path=os.path.join(base_best_model_dir, 'full_eval.txt'))
