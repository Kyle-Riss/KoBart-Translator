"""
Utility script to download KorQuAD 1.0 splits via Hugging Face datasets.

KorQuAD 2.0 must be downloaded manually from the official site and
copied into the same directory afterwards.
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def export_split(dataset, split_name: str, output_dir: Path):
    split = dataset[split_name]
    output_file = output_dir / f"korquad1_{split_name}.json"
    split.to_json(output_file, force_ascii=False)
    print(f"[OK] Saved KorQuAD1 {split_name} split -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download KorQuAD 1.0 dataset")
    parser.add_argument(
        "--output",
        default="data/qa/korquad",
        type=str,
        help="Directory to store the exported JSON files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading KorQuAD 1.0 (squad_kor_v1) ...")
    dataset = load_dataset("squad_kor_v1")

    export_split(dataset, "train", output_dir)
    export_split(dataset, "validation", output_dir)

    print("\nKorQuAD 2.0는 공식 사이트에서 JSON을 내려받아 동일한 디렉터리에 추가해 주세요.")


if __name__ == "__main__":
    main()





