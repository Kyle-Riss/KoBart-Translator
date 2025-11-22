"""
Dialogue Summarization 데이터 분석 스크립트
------------------------------------------
데이터 분포와 길이를 분석하여 학습 설정 최적화에 도움을 줍니다.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_dialogue_data(jsonl_path: Path):
    """Dialogue summarization 데이터를 분석합니다."""
    input_lens = []
    target_lens = []
    splits = Counter()
    topic_counter = Counter()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            input_lens.append(len(obj['input']))
            target_lens.append(len(obj['target']))
            splits[obj.get('meta', {}).get('split', 'unknown')] += 1
            topic = obj.get('meta', {}).get('topic', 'unknown')
            if topic:
                topic_counter[topic] += 1
    
    total = len(input_lens)
    print(f"=" * 60)
    print(f"Dialogue Summarization 데이터 분석")
    print(f"=" * 60)
    print(f"\n총 샘플 수: {total:,}")
    print(f"Split 분포: {dict(splits)}")
    
    print(f"\n입력 길이 통계:")
    print(f"  평균: {sum(input_lens)/len(input_lens):.1f} chars")
    print(f"  중앙값: {sorted(input_lens)[len(input_lens)//2]} chars")
    print(f"  최소: {min(input_lens)} chars")
    print(f"  최대: {max(input_lens)} chars")
    print(f"  > 256 chars: {sum(1 for x in input_lens if x > 256):,} ({100*sum(1 for x in input_lens if x > 256)/len(input_lens):.1f}%)")
    print(f"  > 512 chars: {sum(1 for x in input_lens if x > 512):,} ({100*sum(1 for x in input_lens if x > 512)/len(input_lens):.1f}%)")
    print(f"  > 1024 chars: {sum(1 for x in input_lens if x > 1024):,} ({100*sum(1 for x in input_lens if x > 1024)/len(input_lens):.1f}%)")
    
    print(f"\n타겟 길이 통계:")
    print(f"  평균: {sum(target_lens)/len(target_lens):.1f} chars")
    print(f"  중앙값: {sorted(target_lens)[len(target_lens)//2]} chars")
    print(f"  최소: {min(target_lens)} chars")
    print(f"  최대: {max(target_lens)} chars")
    
    print(f"\n토픽 분포 (상위 10개):")
    for topic, count in topic_counter.most_common(10):
        print(f"  {topic}: {count:,} ({100*count/total:.1f}%)")
    
    # 길이별 분포 히스토그램
    print(f"\n입력 길이 분포 (구간별):")
    bins = [0, 100, 200, 256, 300, 400, 512, 1000, float('inf')]
    bin_labels = ['0-100', '100-200', '200-256', '256-300', '300-400', '400-512', '512-1000', '1000+']
    for i in range(len(bins)-1):
        count = sum(1 for x in input_lens if bins[i] <= x < bins[i+1])
        print(f"  {bin_labels[i]}: {count:,} ({100*count/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze dialogue summarization data.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/processed/dialogue_summarization.jsonl"),
        help="Path to dialogue_summarization.jsonl file",
    )
    args = parser.parse_args()
    
    if not args.jsonl.exists():
        print(f"Error: File not found: {args.jsonl}")
        return
    
    analyze_dialogue_data(args.jsonl)


if __name__ == "__main__":
    main()




