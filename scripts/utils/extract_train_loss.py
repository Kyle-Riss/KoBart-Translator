"""
Train Loss만 추출하는 스크립트
"""

import re
import json
from pathlib import Path

# 터미널에서 본 실제 데이터
terminal_log = """
Epoch 1 Train Loss: 0.3463
Epoch 2 Train Loss: 0.2768
Epoch 3 Train Loss: 0.2651
Epoch 4 Train Loss: 0.2554
Epoch 5 Train Loss: 0.2487
Epoch 6 Train Loss: 0.2447
Epoch 7 Train Loss: 0.2417
Epoch 8 Train Loss: 0.2394
Epoch 9 Train Loss: 0.2377
Epoch 10 Train Loss: 0.2356
Epoch 11 Train Loss: 0.2337
Epoch 12 Train Loss: 0.2447
"""

# 패턴으로 추출
pattern = r'Epoch (\d+) Train Loss: ([\d.]+)'
matches = re.findall(pattern, terminal_log)

if matches:
    print("Epoch | Train Loss")
    print("-" * 30)
    
    epochs = []
    losses = []
    
    for epoch_str, loss_str in matches:
        epoch = int(epoch_str)
        loss = float(loss_str)
        epochs.append(epoch)
        losses.append(loss)
        print(f"{epoch:5d} | {loss:.4f}")
    
    print(f"\n총 {len(epochs)} epochs")
    print(f"최소: {min(losses):.4f} (epoch {epochs[losses.index(min(losses))]})")
    print(f"최대: {max(losses):.4f} (epoch {epochs[losses.index(max(losses))]})")
    print(f"최종: {losses[-1]:.4f} (epoch {epochs[-1]})")
    
    # CSV 저장
    csv_path = Path("runs/train_loss_only.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss\n")
        for epoch, loss in zip(epochs, losses):
            f.write(f"{epoch},{loss:.6f}\n")
    print(f"\n✓ CSV 저장: {csv_path}")
    
    # JSON 저장
    json_path = Path("runs/train_loss_only.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "epochs": epochs,
            "train_loss": losses
        }, f, indent=2)
    print(f"✓ JSON 저장: {json_path}")
else:
    print("Train loss 데이터를 찾을 수 없습니다.")




