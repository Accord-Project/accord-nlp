# Created by Hansi at 28/06/2023

def sweep_config_to_sweep_values(sweep_config):
    """
    Converts an instance of wandb.Config to plain values map.

    wandb.Config varies across versions quite significantly,
    so we use the `keys` method that works consistently.
    """

    return {key: sweep_config[key] for key in sweep_config.keys()}


def load_hf_dataset(data, tokenizer, args, multi_label):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
        )
    else:
        dataset = HFDataset.from_pandas(data)

    if args.labels_map and not args.regression:
        dataset = dataset.map(lambda x: map_labels_to_numeric(x, multi_label, args))

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x, tokenizer=tokenizer, max_seq_length=args.max_seq_length
        ),
        batched=True,
    )

    if args.model_type in ["bert", "xlnet", "albert", "layoutlm"]:
        dataset.set_format(
            type="pt",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )
    else:
        dataset.set_format(type="pt", columns=["input_ids", "attention_mask", "labels"])

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset