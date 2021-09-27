#!/usr/bin/env python -W ignore::UserWarning

import json
import torch
import warnings
import argparse
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from featurize import cache_from_data_file, featurize_files
from data_classes import FactsDataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)
from transformers import (
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)


# avoid torch dummy warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str,
        help="Directory containing split data files"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Output directory to store the model"
    )
    parser.add_argument(
        "--config", type=str, required=False, default="config.yaml",
        help="Config file with training parameters (default: config.yaml)"
    )
    parser.add_argument(
        "--model_name", type=str, required=False,
        default="bertin-project/bertin-roberta-base-spanish",
        help="Name of the model to use"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite previous features"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Whether to train the model"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Whether to evaluate the model"
    )
    parser.add_argument(
        "--hypersearch", action="store_true",
        help="Whether to search for hyperparameters during train"
    )
    return parser.parse_known_args()


def compute_metrics(model_preds, print_results=False):
    # report: accuracy, f1, micro/macro averages
    preds = np.argmax(model_preds.predictions, axis=1)
    labels = model_preds.label_ids
    if print_results:
        print(classification_report(labels, preds))

    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "macro": f1_score(labels, preds, average="macro"),
        "micro": f1_score(labels, preds, average="micro"),
    }


def get_trainer(
    model,
    params,
    model_init=None,
    hypersearch=False,
    train_dataset=None,
    dev_dataset=None,
    output_dir=None
):
    training_args = TrainingArguments(
        **params,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        run_name="newtral-training",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init if hypersearch else None
    )
    
    if hypersearch:
        from hypersearch import HyperSearch
        search = HyperSearch()
        trainer.hyperparameter_search(
            backend=search.backend,
            direction=search.direction,
            hp_space=search.get_search_space,
            compute_objective=search.compute_objective,
        )

    return trainer


def model_init_wrapper(model_name):
    def model_init():
        return RobertaForSequenceClassification.from_pretrained(model_name)

    return model_init


def save_model(model_name, trainer, output_dir, overwrite):
    output_path = Path(output_dir)
    model_path = output_path.joinpath(model_name)
    if model_path.exists() and not overwrite:
        raise ValueError(
            "Model path already exists! Pass --overwrite to create it again"
        )

    # creates dir if necessary
    trainer.save_model(model_path)


def load_dataset(tokenizer, data_dir, file):
    data_path = Path(data_dir)
    cache = data_path.joinpath(cache_from_data_file(tokenizer, file))
    if not cache.exists():
        print(f"Cached features for {file} not found, featurizing first...")
        featurize_files(tokenizer, file, data_dir)

    print(f"Loading features from {cache}")
    data_dict = torch.load(cache)
    dataset = FactsDataset(
        features=data_dict["features"],
        labels=data_dict["labels"]
    )
    return dataset


def main(args, config):
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    train_dataset = load_dataset(tokenizer, args.data_dir, config["train_file"])
    dev_dataset = load_dataset(tokenizer, args.data_dir, config["dev_file"])

    train_params = config["params"]
    if args.train:
        do_hypersearch = config.get("hypersearch", False)
        model_init = model_init_wrapper(args.model_name)
        print(f"Training model (hypersearch={do_hypersearch})...")
        trainer = get_trainer(
            model=None if do_hypersearch else model_init(),
            params=train_params,
            model_init=model_init,
            hypersearch=do_hypersearch,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            output_dir=args.output_dir,
        )
        trainer.train()
        save_model(args.model_name, trainer, args.output_dir, args.overwrite)
    else:
        model_dir = Path(args.output_dir).joinpath(args.model_name)
        model_init = model_init_wrapper(model_dir)
        trainer = get_trainer(
            model_init(),
            train_params,
            hypersearch=False,
            output_dir=args.output_dir,
        )

    if args.evaluate:
        print("Evaluating model...")
        test_dataset = load_dataset(tokenizer, args.data_dir, config["test_file"])
        preds = trainer.predict(test_dataset)
        compute_metrics(preds, print_results=True)
        print(json.dumps(preds.metrics, indent=2) + "\n")


if __name__ == "__main__":
    args, unknown_args = parse_args()
    print(f"Loading config from {args.config}")
    file_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(file_config, OmegaConf.from_cli(unknown_args))
    main(args, config)
