"""
Builds task-specific JSONL files for MultiTaskKoBART training.

Each record emitted by this script has the structure:
{
    "task": "<task_name>",
    "input": "<encoder input>",
    "target": "<decoder target>",
    "meta": {... arbitrary metadata ...}
}

Resulting files are stored in data/processed/<task>.jsonl and can be
directly consumed by custom Dataset classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Utility helpers


def dump_jsonl(samples: Iterable[Dict], output_path: Path) -> int:
    """Stream samples into a JSONL file and return the written count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for sample in samples:
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_style_pairs(pair_str: str | None, columns: Sequence[str]) -> List[Tuple[str, str]]:
    """Return list of (source, target) styles."""
    if not pair_str:
        return [
            (src, tgt)
            for src in columns
            for tgt in columns
            if src != tgt
        ]
    pairs = []
    for chunk in pair_str.split(","):
        if ":" not in chunk:
            raise ValueError(f"Invalid style pair '{chunk}', expected 'src:tgt'")
        src, tgt = chunk.split(":", 1)
        if src not in columns or tgt not in columns:
            raise ValueError(f"Unknown style in pair '{chunk}' (available: {columns})")
        pairs.append((src, tgt))
    return pairs


# ---------------------------------------------------------------------------
# Style Transfer (SmileStyle + 기존 병렬 데이터)


STYLE_DATASET = DATA_ROOT / "style_transfer" / "korean_smile_style_dataset" / "smilestyle_dataset.tsv"
STYLE_DEV_INTERVAL = 20  # roughly 5% dev split


def generate_style_transfer(style_pair_spec: str | None = None) -> Iterator[Dict]:
    """Produce style transfer samples from the Smilegate SmileStyle dataset."""
    dataset_path = STYLE_DATASET
    if not dataset_path.exists():
        print(f"[WARN] Missing SmileStyle dataset at {dataset_path}")
        return iter(())

    with dataset_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        columns = reader.fieldnames or []
        if not columns:
            return iter(())
        pairs = parse_style_pairs(style_pair_spec, columns)

        for row_idx, row in enumerate(reader):
            split_flag = "dev" if STYLE_DEV_INTERVAL and (row_idx % STYLE_DEV_INTERVAL == 0) else "train"
            for src, tgt in pairs:
                src_text = (row.get(src) or "").strip()
                tgt_text = (row.get(tgt) or "").strip()
                if not src_text or not tgt_text:
                    continue
                yield {
                    "task": "style_transfer",
                    "input": f"[style_transfer][{src}->{tgt}] {src_text}",
                    "target": tgt_text,
                    "meta": {
                        "row": row_idx,
                        "source_style": src,
                        "target_style": tgt,
                        "dataset": "smilestyle",
                        "split": split_flag,
                    },
                }


# ---------------------------------------------------------------------------
# Dialogue Summarization (AI Hub)


def iter_aihub_dialogue_summary_from_dir(json_dir: Path, split_hint: str) -> Iterator[Dict]:
    """폴더에서 직접 JSON 파일 읽기 (ZIP 해제 후)"""
    if not json_dir.exists():
        print(f"[WARN] Summary directory missing: {json_dir}")
        return
    
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in: {json_dir}")
        return
    
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            records = data.get("data", [])
            for record in records:
                dialogue_items = record.get("body", {}).get("dialogue", [])
                summary = record.get("body", {}).get("summary", "")
                if not dialogue_items or not summary:
                    continue
                
                dialogue_id = record.get("header", {}).get("dialogueInfo", {}).get("dialogueID", "")
                topic = record.get("header", {}).get("dialogueInfo", {}).get("topic", "")
                
                input_lines = [
                    f"{item.get('participantID', '')}: {item.get('utterance', '')}"
                    for item in dialogue_items
                    if item.get('utterance')
                ]
                if not input_lines:
                    continue
                    
                input_text = "[dialogue_summarization]\n" + "\n".join(input_lines)
                yield {
                    "task": "dialogue_summarization",
                    "input": input_text,
                    "target": summary,
                    "meta": {
                        "dialogue_id": dialogue_id,
                        "topic": topic,
                        "source": json_path.name,
                        "split": split_hint,
                    },
                }
        except Exception as e:
            print(f"[WARN] Error reading {json_path}: {e}")
            continue


def iter_aihub_dialogue_summary(zip_path: Path, split_hint: str) -> Iterator[Dict]:
    """ZIP 파일에서 읽기 (하위 호환성)"""
    if not zip_path.exists():
        return
    
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if not info.filename.endswith(".json"):
                    continue
                with zf.open(info) as fp:
                    data = json.load(fp)
                records = data.get("data", [])
                for record in records:
                    dialogue_items = record.get("body", {}).get("dialogue", [])
                    summary = record.get("body", {}).get("summary", "")
                    if not dialogue_items or not summary:
                        continue
                    
                    dialogue_id = record.get("header", {}).get("dialogueInfo", {}).get("dialogueID", "")
                    topic = record.get("header", {}).get("dialogueInfo", {}).get("topic", "")
                    
                    input_lines = [
                        f"{item.get('participantID', '')}: {item.get('utterance', '')}"
                        for item in dialogue_items
                        if item.get('utterance')
                    ]
                    if not input_lines:
                        continue
                        
                    input_text = "[dialogue_summarization]\n" + "\n".join(input_lines)
                    yield {
                        "task": "dialogue_summarization",
                        "input": input_text,
                        "target": summary,
                        "meta": {
                            "dialogue_id": dialogue_id,
                            "topic": topic,
                            "source": zip_path.name,
                            "split": split_hint,
                        },
                    }
    except Exception as e:
        print(f"[WARN] Error reading ZIP {zip_path}: {e}")


def generate_dialogue_summaries() -> Iterator[Dict]:
    base = DATA_ROOT / "dialogue_summarization" / "aihub_dialogue_summary"
    
    # ZIP 해제된 폴더 우선 시도
    train_dir = base / "Training" / "[라벨]한국어대화요약_train"
    valid_dir = base / "Validation" / "[라벨]한국어대화요약_valid"
    
    if train_dir.exists() and valid_dir.exists():
        # 폴더에서 직접 읽기
        yield from iter_aihub_dialogue_summary_from_dir(train_dir, "train")
        yield from iter_aihub_dialogue_summary_from_dir(valid_dir, "dev")
    else:
        # ZIP 파일 시도 (하위 호환성)
        train_zip = base / "Training" / "[라벨]한국어대화요약_train.zip"
        valid_zip = base / "Validation" / "[라벨]한국어대화요약_valid.zip"
        yield from iter_aihub_dialogue_summary(train_zip, "train")
        yield from iter_aihub_dialogue_summary(valid_zip, "dev")


# ---------------------------------------------------------------------------
# Role-conditioned Generation


CALLER_TOKEN = "[신고자]"
AGENT_TOKEN = "[상담원]"


def _iter_numbered_dialogues(txt_path: Path) -> Iterator[List[str]]:
    if not txt_path.exists():
        print(f"[WARN] Role dialogue file missing: {txt_path}")
        return
    conversation: List[str] = []
    current_idx = None
    with txt_path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            idx, utterance = line.split("\t", 1)
            idx = idx.strip().lstrip("\ufeff")
            utterance = utterance.strip()
            if idx == "1" and conversation:
                yield conversation
                conversation = []
            conversation.append(utterance)
            current_idx = idx
    if conversation:
        yield conversation


def _conversation_to_samples(turns: List[str], domain: str) -> Iterator[Dict]:
    history: List[str] = []
    for turn_idx, utterance in enumerate(turns):
        speaker = CALLER_TOKEN if turn_idx % 2 == 0 else AGENT_TOKEN
        history.append(f"{speaker}: {utterance}")
        if speaker == AGENT_TOKEN:
            context = "\n".join(history[:-1])
            prompt = f"[role_generation]{AGENT_TOKEN}\n{context}"
            yield {
                "task": "role_generation",
                "input": prompt.strip(),
                "target": utterance,
                "meta": {
                    "domain": domain,
                    "turn_index": turn_idx,
                },
            }


ROLE_DEV_INTERVAL = 25  # ~4% dev samples
QA_DEV_INTERVAL = 50    # fallback dev sampling when official dev set missing


def generate_role_data() -> Iterator[Dict]:
    role_root = DATA_ROOT / "role_generation" / "aihub_dialogue_role_dataset"
    sample_idx = 0

    def annotate(sample: Dict) -> Dict:
        nonlocal sample_idx
        if "meta" not in sample:
            sample["meta"] = {}
        sample["meta"]["split"] = (
            "dev" if ROLE_DEV_INTERVAL and (sample_idx % ROLE_DEV_INTERVAL == 0) else "train"
        )
        sample_idx += 1
        return sample

    # Emergency & office txt files
    for txt_name, domain in [
        ("KETI_대화데이터_응급상황.txt", "emergency"),
        ("KETI_대화데이터_일상_오피스.txt", "office"),
    ]:
        path = role_root / txt_name
        for conversation in _iter_numbered_dialogues(path):
            for sample in _conversation_to_samples(conversation, domain=domain):
                yield annotate(sample)

    # JSON office dataset
    json_dir = role_root / "대화데이터_오피스(JSON)"
    if json_dir.exists():
        for json_path in sorted(json_dir.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as fp:
                    records = json.load(fp)
                
                if not isinstance(records, list):
                    records = [records] if records else []
                
                grouped: Dict[int, List[Dict]] = defaultdict(list)
                for entry in records:
                    if isinstance(entry, dict) and "index" in entry:
                        grouped[entry["index"]].append(entry)
                
                for idx, turns in grouped.items():
                    linear_turns: List[str] = []
                    domain = "office"
                    for segment in turns:
                        user_u = (segment.get("user_utterance") or "").strip()
                        system_u = (segment.get("system_utterance") or "").strip()
                        if segment.get("domain"):
                            domain = segment.get("domain", "office")
                        if user_u and user_u.lower() != "null":
                            linear_turns.append(user_u)
                        if system_u and system_u.lower() != "null":
                            linear_turns.append(system_u)
                    if linear_turns:
                        for sample in _conversation_to_samples(
                            linear_turns, domain=domain
                        ):
                            yield annotate(sample)
            except Exception as e:
                print(f"[WARN] Error reading {json_path}: {e}")
                continue
    else:
        print(f"[WARN] Role JSON directory missing: {json_dir}")


# ---------------------------------------------------------------------------
# QA Answer Generation (KorQuAD)


def _load_json_records(path: Path):
    """Load either full JSON or line-delimited JSON."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        records = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records


def _iter_korquad_examples(obj) -> Iterator[Dict]:
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_korquad_examples(item)
        return
    if isinstance(obj, dict):
        if {"context", "question"} <= obj.keys():
            yield obj
            return
        if {"context", "qas"} <= obj.keys():
            context = obj["context"]
            for qa in obj.get("qas", []):
                yield {
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": qa.get("answers", []),
                    "id": qa.get("id"),
                }
            return
        if "data" in obj:
            yield from _iter_korquad_examples(obj["data"])
            return
        if "paragraphs" in obj:
            for paragraph in obj["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph.get("qas", []):
                    answers = qa.get("answers", [])
                    yield {
                        "context": context,
                        "question": qa["question"],
                        "answers": answers,
                        "id": qa.get("id"),
                    }
            return
    raise ValueError("Unsupported KorQuAD JSON structure")


def generate_korquad_samples() -> Iterator[Dict]:
    qa_root = DATA_ROOT / "qa"
    sources: List[Path] = []
    sample_idx = 0

    korquad1_dir = qa_root / "korquad"
    if korquad1_dir.exists():
        sources.extend(sorted(korquad1_dir.glob("*.json")))

    if qa_root.exists():
        # KorQuAD 2.1 디렉토리 처리 (KorQuAD_2.1_train_00, KorQuAD_2.1_dev_00 등)
        for subdir in sorted(qa_root.iterdir()):
            if subdir.is_dir() and ("korquad" in subdir.name.lower() or "2.1" in subdir.name):
                sources.extend(sorted(subdir.glob("*.json")))

    if not sources:
        print(f"[WARN] KorQuAD directory missing: {qa_root}")
        return

    for path in sources:
        data = _load_json_records(path)
        split_hint = "dev" if "dev" in path.name.lower() else "train"
        try:
            iterator = _iter_korquad_examples(data)
            for sample in iterator:
                answers = sample.get("answers") or {}
                if isinstance(answers, dict):
                    answer_texts = answers.get("text", [])
                else:
                    answer_texts = [ans.get("text") for ans in answers if ans and ans.get("text")]
                target = (answer_texts[0] if answer_texts else "").strip()
                if not target:
                    continue
                context = sample.get("context", "")
                question = sample.get("question", "")
                if not context or not question:
                    continue
                if split_hint == "dev":
                    split_flag = "dev"
                else:
                    split_flag = (
                        "dev"
                        if QA_DEV_INTERVAL and (sample_idx % QA_DEV_INTERVAL == 0)
                        else "train"
                    )
                sample_idx += 1
                input_text = f"[qa_generation][DOC] {context}\n[Q] {question}"
                yield {
                    "task": "qa_generation",
                    "input": input_text,
                    "target": target,
                    "meta": {
                        "source_file": str(path.relative_to(qa_root)),
                        "id": sample.get("id"),
                        "split": split_flag,
                    },
                }
        except ValueError as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue


# ---------------------------------------------------------------------------
# CLI


def main():
    parser = argparse.ArgumentParser(description="Prepare task datasets for MultiTaskKoBART.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_ROOT / "processed"),
        help="Directory to store generated JSONL files",
    )
    parser.add_argument(
        "--style-pairs",
        type=str,
        default=None,
        help="Comma-separated list like informal:formal,formal:informal (default: all combinations)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    stats["style_transfer"] = dump_jsonl(generate_style_transfer(args.style_pairs), out_dir / "style_transfer.jsonl")
    stats["dialogue_summarization"] = dump_jsonl(generate_dialogue_summaries(), out_dir / "dialogue_summarization.jsonl")
    stats["role_generation"] = dump_jsonl(generate_role_data(), out_dir / "role_generation.jsonl")
    stats["qa_generation"] = dump_jsonl(generate_korquad_samples(), out_dir / "qa_generation.jsonl")

    with (out_dir / "stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)

    print("[DONE] Dataset preparation summary:")
    for task, count in stats.items():
        print(f"  - {task}: {count} samples -> {out_dir}/{task}.jsonl")


if __name__ == "__main__":
    main()

