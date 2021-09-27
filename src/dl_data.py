import argparse
import requests

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url", type=str, help="Url to download the data from"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Output directory to store data"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite previous data file"
    )

    return parser.parse_args()


def maybe_download(url, output_dir, overwrite=False):
    file_name = url.split('/')[-1]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path.joinpath(file_name)
    if output_file_path.exists() and not overwrite:
        raise RuntimeError(
            f"File `{output_file_path}` already exists, pass --overwrite to"
            " download it again"
        )

    print(f"Downloading {url}")
    r = requests.get(url, allow_redirects=True)
    with open(output_file_path, "wb") as f:
        print(f"Writing {output_file_path}")
        f.write(r.content)


if __name__ == '__main__':
    maybe_download(**vars(parse_args()))
