#!/usr/bin/env python
"""
Utility script for training an 8k SentencePiece tokenizer for the tiny student.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Files or directories that contain plain-text corpora.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the model files (model + vocab) will be written.",
    )
    parser.add_argument("--model_prefix", default="tiny_student_spm", help="Output prefix.")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument(
        "--model_type",
        choices=["unigram", "bpe", "word", "char"],
        default="unigram",
    )
    parser.add_argument(
        "--user_defined_symbols",
        nargs="*",
        default=["<dialogue>", "<style>", "<qa>", "<role>"],
        help="Additional special tokens to reserve.",
    )
    return parser.parse_args()


def collect_text_files(sources: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for entry in sources:
        path = Path(entry)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.txt")))
            paths.extend(sorted(path.rglob("*.jsonl")))
        else:
            matches = list(Path().glob(entry))
            paths.extend([p for p in matches if p.is_file()])
    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def train_sentencepiece(
    input_files: List[Path],
    output_dir: Path,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
    model_type: str,
    user_defined_symbols: List[str],
) -> None:
    if not input_files:
        raise ValueError("No input text files resolved for SentencePiece training.")
    output_dir.mkdir(parents=True, exist_ok=True)

    absolute_inputs = ",".join(str(path.resolve()) for path in input_files)
    prefix = output_dir / model_prefix

    spm.SentencePieceTrainer.Train(
        input=absolute_inputs,
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        user_defined_symbols=user_defined_symbols,
        pad_id=1,
        bos_id=0,
        eos_id=2,
        unk_id=3,
    )


def main() -> None:
    args = parse_args()
    files = collect_text_files(args.sources)
    train_sentencepiece(
        input_files=files,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        user_defined_symbols=args.user_defined_symbols,
    )


if __name__ == "__main__":
    main()







