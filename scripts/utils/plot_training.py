"""
학습 로그 시각화 스크립트
------------------------
JSON 로그 파일이나 TensorBoard 로그를 읽어서 loss curve를 그립니다.
"""

import argparse
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트를 설정합니다."""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS에서 사용 가능한 한글 폰트
        korean_fonts = ['AppleGothic', 'NanumGothic', 'Apple SD Gothic Neo']
    elif system == 'Windows':
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'Gulim']
    else:  # Linux
        korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in korean_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            return font_name
    
    # 한글 폰트를 찾지 못한 경우 경고
    print(f"Warning: Korean font not found. Available fonts: {sorted(set(available_fonts))[:10]}")
    plt.rcParams['axes.unicode_minus'] = False
    return None

# 한글 폰트 설정
setup_korean_font()


def plot_from_json(json_path: Path, output_path: Path = None):
    """JSON 로그 파일에서 loss curve를 그립니다."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = data["epochs"]
    train_loss = data["train_loss"]
    dev_loss = data["dev_loss"]
    dev_task_losses = data.get("dev_task_losses", {})

    # Train-Dev gap 계산
    gaps = [abs(t - d) for t, d in zip(train_loss, dev_loss)]
    min_gap_epoch = epochs[gaps.index(min(gaps))] if gaps else None
    min_gap_value = min(gaps) if gaps else None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Overall train/dev loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=2, markersize=4)
    ax1.plot(epochs, dev_loss, label="Dev Loss", marker="s", linewidth=2, markersize=4)
    
    # Best epoch (lowest dev loss)
    if "best_epoch" in data and data["best_epoch"]:
        best_epoch = data["best_epoch"]
        best_loss = data["best_dev_loss"]
        ax1.axvline(x=best_epoch, color="r", linestyle="--", alpha=0.5, label=f"Best Dev (epoch {best_epoch})")
        ax1.scatter([best_epoch], [best_loss], color="r", s=100, zorder=5)
    
    # Min gap epoch (train과 dev가 가장 가까운 지점)
    if min_gap_epoch:
        min_gap_idx = epochs.index(min_gap_epoch)
        ax1.axvline(x=min_gap_epoch, color="g", linestyle="--", alpha=0.5, label=f"Min Gap (epoch {min_gap_epoch})")
        ax1.scatter([min_gap_epoch], [train_loss[min_gap_idx]], color="g", s=100, zorder=5, marker="^")
        ax1.scatter([min_gap_epoch], [dev_loss[min_gap_idx]], color="g", s=100, zorder=5, marker="v")
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Dev Loss (가까울수록 좋음)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Train-Dev Gap 시각화
    ax2 = axes[0, 1]
    ax2.plot(epochs, gaps, label="Train-Dev Gap", color="purple", marker="o", linewidth=2, markersize=4)
    if min_gap_epoch:
        ax2.axvline(x=min_gap_epoch, color="g", linestyle="--", alpha=0.5, label=f"Min Gap (epoch {min_gap_epoch})")
        ax2.scatter([min_gap_epoch], [min_gap_value], color="g", s=100, zorder=5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("|Train Loss - Dev Loss|")
    ax2.set_title("Train-Dev Gap (작을수록 좋음)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Per-task dev losses
    ax3 = axes[1, 0]
    has_tasks = False
    for task, losses in dev_task_losses.items():
        if losses and len(losses) == len(epochs):
            ax3.plot(epochs, losses, label=task, marker="o", linewidth=2, markersize=4)
            has_tasks = True
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Dev Loss")
    ax3.set_title("Dev Loss by Task")
    if has_tasks:
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 통계 요약
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = "학습 통계\n" + "="*30 + "\n\n"
    stats_text += f"총 Epochs: {len(epochs)}\n"
    if "best_epoch" in data and data["best_epoch"]:
        best_epoch = data["best_epoch"]
        best_idx = epochs.index(best_epoch)
        stats_text += f"\nBest Dev Loss:\n"
        stats_text += f"  Epoch: {best_epoch}\n"
        stats_text += f"  Dev Loss: {data['best_dev_loss']:.4f}\n"
        stats_text += f"  Train Loss: {train_loss[best_idx]:.4f}\n"
        stats_text += f"  Gap: {abs(train_loss[best_idx] - data['best_dev_loss']):.4f}\n"
    if min_gap_epoch:
        min_gap_idx = epochs.index(min_gap_epoch)
        stats_text += f"\nMin Gap:\n"
        stats_text += f"  Epoch: {min_gap_epoch}\n"
        stats_text += f"  Train Loss: {train_loss[min_gap_idx]:.4f}\n"
        stats_text += f"  Dev Loss: {dev_loss[min_gap_idx]:.4f}\n"
        stats_text += f"  Gap: {min_gap_value:.4f}\n"
    if train_loss and dev_loss:
        stats_text += f"\n최종 상태:\n"
        stats_text += f"  Train Loss: {train_loss[-1]:.4f}\n"
        stats_text += f"  Dev Loss: {dev_loss[-1]:.4f}\n"
        stats_text += f"  Gap: {gaps[-1]:.4f}\n"
    # 한글 폰트 사용 (monospace 대신)
    current_font = plt.rcParams.get('font.family', 'sans-serif')
    if isinstance(current_font, list):
        current_font = current_font[0] if current_font else 'sans-serif'
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family=current_font, verticalalignment='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ 그래프 저장: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from JSON log file.")
    parser.add_argument("--json", type=Path, required=True, help="JSON log file path.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: show plot).")
    args = parser.parse_args()

    if not args.json.exists():
        print(f"Error: JSON file not found: {args.json}")
        return

    plot_from_json(args.json, args.output)


if __name__ == "__main__":
    main()

