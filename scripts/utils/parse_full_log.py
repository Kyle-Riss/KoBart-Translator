"""
전체 학습 로그 파일을 파싱해서 JSON 형식으로 변환
"""

import re
import json
from pathlib import Path

def parse_log_file(log_path: Path):
    """로그 파일을 읽어서 학습 히스토리를 추출"""
    # 여러 인코딩 시도
    encodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8', 'utf-8-sig', 'latin-1', 'cp949', 'euc-kr']
    log_text = None
    for enc in encodings:
        try:
            with open(log_path, 'r', encoding=enc) as f:
                log_text = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if log_text is None:
        # 바이너리 모드로 읽기
        with open(log_path, 'rb') as f:
            raw = f.read()
        log_text = raw.decode('utf-8', errors='ignore')
    
    epochs = []
    train_losses = []
    dev_losses = []
    dev_task_losses = {
        "dialogue_summarization": [],
        "qa_generation": [],
        "role_generation": [],
        "style_transfer": [],
    }
    
    # Epoch별 train loss 추출
    train_pattern = r"Epoch (\d+) Train Loss: ([\d.]+)"
    train_matches = list(re.finditer(train_pattern, log_text))
    
    # Epoch별 dev loss 추출
    dev_pattern = r"Epoch (\d+) Dev Losses -> (.+)"
    dev_matches = list(re.finditer(dev_pattern, log_text))
    
    # Epoch 번호로 정렬
    all_epochs = set()
    for match in train_matches:
        all_epochs.add(int(match.group(1)))
    for match in dev_matches:
        all_epochs.add(int(match.group(1)))
    
    epochs = sorted(all_epochs)
    
    # Train loss 매핑
    train_dict = {}
    for match in train_matches:
        epoch = int(match.group(1))
        loss = float(match.group(2))
        train_dict[epoch] = loss
    
    # Dev loss 매핑
    dev_dict = {}
    dev_task_dict = {task: {} for task in dev_task_losses.keys()}
    
    for match in dev_matches:
        epoch = int(match.group(1))
        dev_str = match.group(2)
        
        # overall loss 추출
        overall_match = re.search(r"overall: ([\d.]+)", dev_str)
        if overall_match:
            dev_dict[epoch] = float(overall_match.group(1))
        
        # 태스크별 loss 추출
        for task in dev_task_losses.keys():
            task_pattern = rf"{task}: ([\d.]+)"
            task_match = re.search(task_pattern, dev_str)
            if task_match:
                loss_val = float(task_match.group(1))
                dev_task_dict[task][epoch] = loss_val
    
    # Epoch 순서대로 리스트 생성
    train_losses = [train_dict.get(epoch) for epoch in epochs if epoch in train_dict]
    dev_losses = [dev_dict.get(epoch) for epoch in epochs if epoch in dev_dict]
    
    for task in dev_task_losses.keys():
        dev_task_losses[task] = [dev_task_dict[task].get(epoch) for epoch in epochs if epoch in dev_task_dict[task]]
    
    # None 값 제거
    epochs_clean = [e for i, e in enumerate(epochs) if train_losses[i] is not None or (i < len(dev_losses) and dev_losses[i] is not None)]
    train_losses_clean = [l for l in train_losses if l is not None]
    dev_losses_clean = [l for l in dev_losses if l is not None]
    
    # Best epoch 찾기
    best_epoch = None
    best_dev_loss = float('inf')
    for i, loss in enumerate(dev_losses_clean):
        if loss < best_dev_loss:
            best_dev_loss = loss
            best_epoch = epochs_clean[i] if i < len(epochs_clean) else None
    
    result = {
        "epochs": epochs_clean[:len(train_losses_clean)],
        "train_loss": train_losses_clean,
        "dev_loss": dev_losses_clean,
        "dev_task_losses": {k: v for k, v in dev_task_losses.items() if any(x is not None for x in v)},
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss if best_epoch else None,
    }
    
    return result

if __name__ == "__main__":
    log_path = Path("logs/train_output.log")
    
    if not log_path.exists():
        print(f"Error: 로그 파일을 찾을 수 없습니다: {log_path}")
        exit(1)
    
    result = parse_log_file(log_path)
    
    output_path = Path("runs/training_log_full.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 로그 파싱 완료: {output_path}")
    print(f"  총 Epochs: {len(result['epochs'])}")
    print(f"  Train losses: {len(result['train_loss'])}")
    print(f"  Dev losses: {len(result['dev_loss'])}")
    if result['best_epoch']:
        print(f"  Best epoch: {result['best_epoch']}, Best dev loss: {result['best_dev_loss']:.4f}")
    
    # 태스크별 통계
    print(f"\n태스크별 Dev Loss:")
    for task, losses in result['dev_task_losses'].items():
        if losses:
            print(f"  {task}: {len(losses)} epochs, 최종: {losses[-1]:.4f}")

