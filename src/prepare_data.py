import random
import argparse
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=str, help="Initial data to split"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=False, default=None,
        help="Output directory to store splits (default dirname `input_file`)"
    )
    parser.add_argument(
        "--dev_size", type=float, required=False, default=0.0,
        help="Percentage of data to use for dev split, typically 0.2."
        " By default it will not be created (=0.0)"
    )
    parser.add_argument(
        "--test_size", type=float, required=True,
        help="Percentage of data to use for test split, typically 0.2"
    )
    parser.add_argument(
        "--random_state", type=int, required=False, default=None,
        help="Random state to initialize the splitter with (use for "
        "reproducibility)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite previous split files"
    )

    return parser.parse_args()


def split_data(X, y, test_size, random_state=0):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    return [(train, test) for train, test in splitter.split(X, y)][0]


def get_splits(input_path, dev_size, test_size, random_state=0):
    data = pd.read_csv(input_path)
    total_len = len(data)
    dev_test_size = dev_size + test_size
    train_index, dev_test_index = split_data(
        data["text"], data["claim"],
        test_size=test_size, random_state=random_state
    )

    if dev_size > 0:
        final_dev_size = (dev_size * total_len) / (dev_test_size * total_len)
        dev_index, test_index = split_data(
            data["text"][dev_test_index], data["claim"][dev_test_index],
            test_size=final_dev_size, random_state=random_state
        )
    else:
        test_index = dev_test_index
        dev_index = None

    train_split = pd.concat(
        [data["text"][train_index], data["claim"][train_index]], axis=1
    )
    dev_split = None
    if dev_index is not None:
        dev_split = pd.concat(
            [data["text"][dev_index], data["claim"][dev_index]], axis=1
        )

    test_split = pd.concat(
        [data["text"][test_index], data["claim"][test_index]], axis=1
    )

    return train_split, dev_split, test_split


def maybe_split(
    input_file, output_dir,
    dev_size=0.0, test_size=0.20,
    random_state=0, overwrite=False
):
    # setup
    input_path = Path(input_file)
    output_path = Path(output_dir)
    train_path = output_path.joinpath("train.csv")
    dev_path = output_path.joinpath("dev.csv")
    test_path = output_path.joinpath("test.csv")

    # I/O checks
    if not input_path.exists():
        raise RuntimeError(
            "Input file does not exist! You forgot to download it first?"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    if train_path.exists() and not overwrite:
        raise ValueError(
            "Train split already exists! Pass --overwrite to create it again"
        )

    if test_path.exists() and not overwrite:
        raise ValueError(
            "Test split already exists! Pass --overwrite to create it again"
        )

    if dev_size > 0 and dev_path.exists() and not overwrite:
        raise ValueError(
            "Dev split already exists! Pass --overwrite to create it again"
        )

    # data work
    splits = get_splits(
        input_path, dev_size, test_size, random_state=random_state
    )

    # save everything
    for split_df, path in zip(splits, [train_path, dev_path, test_path]):
        if split_df is not None:
            print(f"Saved split to {path}")
            split_df.to_csv(path)


if __name__ == "__main__":
    args = parse_args()
    # randomize on each run
    if args.random_state is None:
        args.random_state = random.randint(0, 10000)

    if args.output_dir is None:
        args.output_dir = str(Path(args.input_file).absolute().parent)

    maybe_split(**vars(args))
