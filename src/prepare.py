import random
import unicodedata
from pathlib import Path
import pandas as pd

from more_itertools import divide, flatten
from tap import Tap
from tqdm import tqdm

import utils


class Args(Tap):
    input_file_path: Path = "./data/dataset.csv"
    text_column: str = "text"
    label_column: str = "label"

    output_dir: Path = "./datasets/"
    seed: int = 42


# 本文の前処理
# 重複した改行の削除、文頭の全角スペースの削除、NFKC正規化を実施
def process_body(body: list[str]) -> str:
    body = [unicodedata.normalize("NFKC", line) for line in body]
    body = [line.strip("　").strip() for line in body]
    body = [line for line in body if line]
    body = "\n".join(body)
    return body


def main(args: Args):
    random.seed(args.seed)

    data = []
    labels = set()

    df = pd.read_csv(args.input_file_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[args.text_column]
        label = row[args.label_column]
        labels.add(label)

        data.append(
            {
                "label": label,
                "text": text,
            }
        )

    random.shuffle(data)

    utils.save_jsonl(data, args.output_dir / "all.jsonl")
    utils.save_json(list(labels), args.output_dir / "labels.json")

    portions = list(divide(10, data))
    train, val, test = list(flatten(portions[:-2])), portions[-2], portions[-1]
    utils.save_jsonl(train, args.output_dir / "train.jsonl")
    utils.save_jsonl(val, args.output_dir / "val.jsonl")
    utils.save_jsonl(test, args.output_dir / "test.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
