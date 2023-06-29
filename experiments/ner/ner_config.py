# Created by Hansi at 28/06/2023
import os
from multiprocessing import cpu_count

TEMP_DIRECTORY = "temp"
SEED = 157


ner_args = {
    'output_dir': os.path.join(TEMP_DIRECTORY, "outputs"),
    'best_model_dir':  os.path.join(TEMP_DIRECTORY, "outputs/best_model"),
    'cache_dir': os.path.join(TEMP_DIRECTORY, "cache_dir"),

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,  # 256, 128 (max-104)
    'train_batch_size': 8,  # 32
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 512,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,
    'n_fold': 1,

    'logging_steps': 8,  # 20
    'save_steps': 8,  # 20
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': False,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": True,
    'evaluate_during_training_steps': 8,  # 20
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': False,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': "ACCORD-NLP",
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "config": {},
    "local_rank": -1,
    "encoding": None,

}
