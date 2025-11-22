"""
터미널 학습 로그를 파싱해서 JSON 형식으로 변환
"""

import re
import json
from pathlib import Path

def parse_terminal_log(log_text: str):
    """터미널 출력 로그를 파싱해서 학습 히스토리를 추출"""
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
    for match in re.finditer(train_pattern, log_text):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        if epoch not in epochs:
            epochs.append(epoch)
            train_losses.append(loss)
    
    # Epoch별 dev loss 추출
    dev_pattern = r"Epoch (\d+) Dev Losses -> (.+)"
    for match in re.finditer(dev_pattern, log_text):
        epoch = int(match.group(1))
        dev_str = match.group(2)
        
        # overall loss 추출
        overall_match = re.search(r"overall: ([\d.]+)", dev_str)
        if overall_match:
            if epoch in epochs:
                idx = epochs.index(epoch)
                if idx < len(dev_losses):
                    dev_losses[idx] = float(overall_match.group(1))
                else:
                    dev_losses.append(float(overall_match.group(1)))
        
        # 태스크별 loss 추출
        for task in dev_task_losses.keys():
            task_pattern = rf"{task}: ([\d.]+)"
            task_match = re.search(task_pattern, dev_str)
            if task_match:
                loss_val = float(task_match.group(1))
                if epoch in epochs:
                    idx = epochs.index(epoch)
                    while len(dev_task_losses[task]) <= idx:
                        dev_task_losses[task].append(None)
                    dev_task_losses[task][idx] = loss_val
    
    # None 값 제거 및 정렬
    for task in dev_task_losses:
        dev_task_losses[task] = [x for x in dev_task_losses[task] if x is not None]
    
    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "dev_loss": dev_losses,
        "dev_task_losses": {k: v for k, v in dev_task_losses.items() if v},
    }

if __name__ == "__main__":
    # 터미널 로그에서 본 데이터를 직접 입력
    log_data = """
Epoch 1 Train Loss: 0.3463
Epoch 1 Dev Losses -> dialogue_summarization: 0.5875, qa_generation: 0.1243, role_generation: 0.2278, style_transfer: 0.4193, overall: 0.3397

Epoch 2 Train Loss: 0.2768
Epoch 2 Dev Losses -> dialogue_summarization: 0.5828, qa_generation: 0.1276, role_generation: 0.2217, style_transfer: 0.4371, overall: 0.3423

Epoch 3 Train Loss: 0.2651
Epoch 3 Dev Losses -> dialogue_summarization: 0.5912, qa_generation: 0.1338, role_generation: 0.2204, style_transfer: 0.4440, overall: 0.3473

Epoch 4 Train Loss: 0.2554
Epoch 4 Dev Losses -> dialogue_summarization: 0.5962, qa_generation: 0.1389, role_generation: 0.2193, style_transfer: 0.4472, overall: 0.3504

Epoch 5 Train Loss: 0.2487
Epoch 5 Dev Losses -> dialogue_summarization: 0.6053, qa_generation: 0.1429, role_generation: 0.2208, style_transfer: 0.4623, overall: 0.3578

Epoch 6 Train Loss: 0.2447
Epoch 6 Dev Losses -> dialogue_summarization: 0.6114, qa_generation: 0.1455, role_generation: 0.2211, style_transfer: 0.4696, overall: 0.3619

Epoch 7 Train Loss: 0.2417
Epoch 7 Dev Losses -> dialogue_summarization: 0.6106, qa_generation: 0.1509, role_generation: 0.2240, style_transfer: 0.4830, overall: 0.3671

Epoch 8 Train Loss: 0.2394
Epoch 8 Dev Losses -> dialogue_summarization: 0.6179, qa_generation: 0.1526, role_generation: 0.2197, style_transfer: 0.4942, overall: 0.3711

Epoch 9 Train Loss: 0.2377
Epoch 9 Dev Losses -> dialogue_summarization: 0.6241, qa_generation: 0.1596, role_generation: 0.2234, style_transfer: 0.4961, overall: 0.3758

Epoch 10 Train Loss: 0.2356
Epoch 10 Dev Losses -> dialogue_summarization: 0.6238, qa_generation: 0.1583, role_generation: 0.2259, style_transfer: 0.5021, overall: 0.3775

Epoch 11 Train Loss: 0.2337
Epoch 11 Dev Losses -> dialogue_summarization: 0.6274, qa_generation: 0.1637, role_generation: 0.2252, style_transfer: 0.5149, overall: 0.3828
"""
    
    result = parse_terminal_log(log_data)
    result["best_epoch"] = 1  # overall loss가 가장 낮은 epoch
    result["best_dev_loss"] = result["dev_loss"][0] if result["dev_loss"] else None
    
    output_path = Path("runs/training_log_from_terminal.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 로그 파싱 완료: {output_path}")
    print(f"  Epochs: {len(result['epochs'])}")
    print(f"  Train losses: {len(result['train_loss'])}")
    print(f"  Dev losses: {len(result['dev_loss'])}")
    print(f"  Best epoch: {result['best_epoch']}, Best dev loss: {result['best_dev_loss']:.4f}")




