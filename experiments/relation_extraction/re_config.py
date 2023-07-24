# Created by Hansi at 21/07/2023
import os
from multiprocessing import cpu_count

TEMP_DIRECTORY = "temp"
SEED = 157


re_args = {
    'output_dir': os.path.join(TEMP_DIRECTORY, "outputs"),
    "best_model_dir": os.path.join("outputs/best_model"),
    'cache_dir': os.path.join(TEMP_DIRECTORY, "cache_dir"),

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'train_batch_size': 2,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 1,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,
    'n_fold': 1,

    'logging_steps': 8,
    'save_steps': 8,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': False,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": False,
    'evaluate_during_training_steps': 8,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    # 'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'process_count': 1,
    # 'n_gpu': 1,
    'n_gpu': 0,
    'use_multiprocessing': False,
    # "multiprocessing_chunksize": 500,
    "multiprocessing_chunksize": -1,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    # "tagging": True,
    # "begin_tag": "<begin>",
    # "end_tag": "<end>",
    # "merge_type": "cls",  # "cls, "concat", "add", "avg", "entity-pool", "entity-first", "entity-last", "cls-*"
    "special_tags": ["<e1>", "<e2>"],  # Should be either begin_tag or end_tag
    # Need to be provided only for the merge_types: concat, add and avg. For others this will be automatically set.

    "manual_seed": SEED,

    "config": {},
    "local_rank": -1,
    "encoding": None,
}





