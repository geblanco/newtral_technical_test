import torch
import argparse
import pandas as pd

from pathlib import Path
from transformers import RobertaTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str,
        help="Directory containing split data files"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=False, default=None,
        help="Output directory to store features (default to `data_dir`)"
    )
    parser.add_argument(
        "--model_name", type=str, required=False,
        default="bertin-project/bertin-roberta-base-spanish",
        help="Name of the model to use"
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", required=True,
        choices=["train", "dev", "test"],
        help="Splits to featurize"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite previous features"
    )
    return parser.parse_args()


def read_data_split(split_path):
    data = pd.read_csv(split_path)
    return data["text"].to_list(), data["claim"].to_list()


def cache_from_data_file(tokenizer, data_file):
    file_name = Path(data_file).absolute().name
    return f"cached_{file_name}_{tokenizer.__class__.__name__}"


def featurize_files(tokenizer, files, output_dir, overwrite):
    files = files if isinstance(files, (list, tuple)) else [files]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for data_path in map(Path, files):
        if not data_path.exists():
            raise RuntimeError(
                f"{data_path} not found! Maybe you forgot to split the data?"
            )

        features_name = cache_from_data_file(tokenizer, data_path)
        features_file = output_path.joinpath(features_name)
        if features_file.exists() and not overwrite:
            raise ValueError(
                f"{features_file} already exists! Pass --overwrite to create "
                "it again"
            )

        texts, labels = read_data_split(data_path)
        print(f"Creating features for {data_path}")
        # ToDo :=
        # - add max_length
        # - check overflow tokens
        features = tokenizer(
            texts, truncation=True, padding=True, max_length=512
        )
        print(f"Saving features to {features_file}")
        torch.save(dict(features=features, labels=labels), features_file)


def main(data_dir, output_dir, model_name, splits, overwrite):
    data_path = Path(data_dir)
    output_dir = data_path if output_dir is None else output_dir
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    split_paths = [data_path.joinpath(f"{split}.csv") for split in splits]
    featurize_files(tokenizer, split_paths, output_dir, overwrite)


if __name__ == "__main__":
    main(**vars(parse_args()))
