"""
Train Loss만 시각화하는 스크립트
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_train_loss(json_path: Path, output_path: Path = None):
    """Train loss만 시각화"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Train loss 플롯
    ax.plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=2, markersize=6, color="blue")
    
    # 최소값 표시
    min_idx = train_loss.index(min(train_loss))
    min_epoch = epochs[min_idx]
    min_loss = train_loss[min_idx]
    ax.scatter([min_epoch], [min_loss], color="green", s=150, zorder=5, marker="*", label=f"Min (epoch {min_epoch})")
    ax.annotate(f"Min: {min_loss:.4f}\n(epoch {min_epoch})", 
                xy=(min_epoch, min_loss), 
                xytext=(10, 10), 
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    
    # 최종값 표시
    final_epoch = epochs[-1]
    final_loss = train_loss[-1]
    ax.scatter([final_epoch], [final_loss], color="red", s=150, zorder=5, marker="s", label=f"Final (epoch {final_epoch})")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Train Loss Over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 통계 텍스트 추가
    stats_text = f"Total Epochs: {len(epochs)}\n"
    stats_text += f"Min Loss: {min_loss:.4f} (epoch {min_epoch})\n"
    stats_text += f"Max Loss: {max(train_loss):.4f} (epoch {epochs[train_loss.index(max(train_loss))]})\n"
    stats_text += f"Final Loss: {final_loss:.4f} (epoch {final_epoch})\n"
    stats_text += f"Improvement: {train_loss[0] - final_loss:.4f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ 그래프 저장: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    json_path = Path("runs/train_loss_only.json")
    
    if not json_path.exists():
        print(f"Error: JSON 파일을 찾을 수 없습니다: {json_path}")
        exit(1)
    
    output_path = Path("runs/train_loss_plot.png")
    plot_train_loss(json_path, output_path)




