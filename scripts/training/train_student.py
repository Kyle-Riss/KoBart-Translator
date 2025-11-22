#!/usr/bin/env python
"""
Knowledge distillation training loop for the tiny student model.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from kobart_translator.tiny_student import TinyStudentConfig, TinyStudentForConditionalGeneration


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    source: str
    target: str
    task: str


class JSONLinesDataset(Dataset):
    """
    Simple dataset that expects {"input": "...", "target": "...", "task": "..."} lines.
    """

    def __init__(self, paths: List[Path]):
        self.samples: List[Sample] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self.samples.append(
                        Sample(
                            source=record["input"],
                            target=record["target"],
                            task=record.get("task", "style_transfer"),
                        )
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = input_ids[:, :-1]
    shifted[shifted == -100] = pad_token_id
    return shifted


def collate_fn(batch: List[Sample], tokenizer) -> Dict[str, torch.Tensor]:
    sources = [item.source for item in batch]
    targets = [item.target for item in batch]
    tasks = [item.task for item in batch]

    inputs = tokenizer(
        sources,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    target_encodings = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    labels = target_encodings["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    decoder_input_ids = shift_tokens_right(
        labels.clone(),
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "tasks": tasks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny student with KD.")
    parser.add_argument("--teacher", type=str, default="cosmoquester/bart-ko-small")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="SentencePiece tokenizer dir.")
    parser.add_argument("--data_paths", nargs="+", type=Path, default=[Path("data/processed/style_transfer.jsonl")])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.7, help="KD soft target weight.")
    parser.add_argument("--beta", type=float, default=0.2, help="Label CE weight.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Hidden-state MSE weight.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--save_dir", type=Path, default=Path("runs/tiny_student"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    dataset = JSONLinesDataset([Path(p) for p in args.data_paths])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        args.teacher,
        output_hidden_states=True,
    ).to(args.device)
    teacher.eval()

    student_config = TinyStudentConfig()
    student = TinyStudentForConditionalGeneration(student_config).to(args.device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            decoder_input_ids = batch["decoder_input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            logits_student = []
            enc_states_student = []
            for idx in range(input_ids.size(0)):
                outputs = student(
                    input_ids=input_ids[idx : idx + 1],
                    attention_mask=attention_mask[idx : idx + 1],
                    decoder_input_ids=decoder_input_ids[idx : idx + 1],
                    task=batch["tasks"][idx],
                )
                logits_student.append(outputs["logits"])
                enc_states_student.append(outputs["encoder_hidden_states"])

            student_logits = torch.cat(logits_student, dim=0)
            student_enc = torch.cat(enc_states_student, dim=0)
            teacher_logits = teacher_outputs.logits
            teacher_enc = teacher_outputs.encoder_last_hidden_state

            kd_loss = F.kl_div(
                F.log_softmax(student_logits / args.temperature, dim=-1),
                F.softmax(teacher_logits / args.temperature, dim=-1),
                reduction="batchmean",
            ) * (args.temperature**2)
            ce_loss = ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
            hidden_mse = F.mse_loss(student_enc, teacher_enc.detach())
            loss = args.alpha * kd_loss + args.beta * ce_loss + args.gamma * hidden_mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            if global_step % 50 == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} kd={kd_loss.item():.4f} "
                    f"ce={ce_loss.item():.4f} mse={hidden_mse.item():.4f}"
                )
            global_step += 1

        ckpt_path = args.save_dir / f"student_epoch{epoch}.pt"
        torch.save(
            {
                "model_state": student.state_dict(),
                "config": student.student_config.__dict__,
                "tokenizer": args.tokenizer_path,
            },
            ckpt_path,
        )
        print(f"[+] Saved checkpoint to {ckpt_path}")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()


