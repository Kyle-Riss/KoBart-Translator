"""
실제 데이터로 스타일 변환 학습
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from kobart_translator import MultiTaskKoBART, StyleTransferDataLoader
from tqdm import tqdm
import os


class StyleTransferDataset(Dataset):
    """스타일 변환 데이터셋"""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 입력에 스타일 정보 추가
        input_text = f"[{item['source_style']}→{item['target_style']}] {item['input']}"
        
        # 토큰화
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            item['target'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'task': 'style_transfer',
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0)
        }


def train_step(model, batch, optimizer, device, config, tokenizer):
    """한 스텝 학습"""
    model.train()
    
    # 데이터를 디바이스로 이동
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Decoder input ids 생성
    decoder_input_ids = labels.clone()
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = config.decoder_start_token_id
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        task='style_transfer'
    )
    
    # Loss 계산
    logits = outputs['logits']
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def evaluate(model, dataloader, device, config, tokenizer):
    """모델 평가"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Decoder input ids 생성
            decoder_input_ids = labels.clone()
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = config.decoder_start_token_id
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                task='style_transfer'
            )
            
            # Loss 계산
            logits = outputs['logits']
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_epoch(model, dataloader, optimizer, device, config, epoch, tokenizer):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        loss = train_step(model, batch, optimizer, device, config, tokenizer)
        total_loss += loss
        num_batches += 1
        
        # Progress bar 업데이트
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def main():
    """메인 학습 함수"""
    print("="*60)
    print("스타일 변환 모델 학습")
    print("="*60)
    print()
    
    # 하이퍼파라미터
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 20
    MAX_LENGTH = 128
    SAVE_DIR = "checkpoints"
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 디렉토리 생성
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 토크나이저 로드
    print("\n토크나이저 로딩 중...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    print("✓ 토크나이저 로드 완료")
    
    # 모델 로드
    print("\n모델 로딩 중...")
    model = MultiTaskKoBART()
    model.to(device)
    print("✓ 모델 로드 완료")
    
    # 인코더 고정 옵션 (선택사항)
    # model.freeze_encoder()
    # print("✓ 인코더 고정 (디코더만 학습)")
    
    # 데이터 로드
    print("\n데이터 로딩 중...")
    data_loader = StyleTransferDataLoader()
    
    # 학습 데이터 (전체의 1/20로 축소)
    train_data = data_loader.get_train_data(
        use_parallel=True,
        use_opus=True,  # OPUS 데이터 포함
        opus_samples=2500,  # OPUS 샘플 수 감소 (스타일당 833개 x 3 = 2500개)
        parallel_samples=3500  # 병렬 데이터 샘플 수 (약 70182/20 = 3500개)
    )
    print(f"✓ 학습 데이터: {len(train_data):,}개 (전체의 1/20)")
    
    # 검증 데이터
    dev_data = data_loader.get_dev_data()
    print(f"✓ 검증 데이터: {len(dev_data):,}개")
    
    # 데이터셋 생성
    train_dataset = StyleTransferDataset(train_data, tokenizer, MAX_LENGTH)
    dev_dataset = StyleTransferDataset(dev_data, tokenizer, MAX_LENGTH)
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 학습 정보 출력
    print(f"\n학습 설정:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Max length: {MAX_LENGTH}")
    print(f"  - Total steps: {len(train_dataloader) * NUM_EPOCHS:,}")
    
    # 학습 시작
    print("\n" + "="*60)
    print("학습 시작")
    print("="*60)
    
    best_dev_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")
        
        # 학습
        train_loss = train_epoch(
            model, train_dataloader, optimizer, device,
            model.config, epoch, tokenizer
        )
        
        # 검증
        print("검증 중...")
        dev_loss = evaluate(model, dev_dataloader, device, model.config, tokenizer)
        
        print(f"\nEpoch {epoch} 결과:")
        print(f"  - Train Loss: {train_loss:.4f}")
        print(f"  - Dev Loss: {dev_loss:.4f}")
        
        # 모델 저장
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_path = os.path.join(SAVE_DIR, f"best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'dev_loss': dev_loss,
            }, save_path)
            print(f"✓ 최고 성능 모델 저장: {save_path}")
        
        # 체크포인트 저장
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'dev_loss': dev_loss,
        }, checkpoint_path)
        print(f"✓ 체크포인트 저장: checkpoint_epoch_{epoch}.pt")
    
    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)
    print(f"\n최고 검증 손실: {best_dev_loss:.4f}")
    print(f"모델 저장 위치: {SAVE_DIR}")
    
    # 학습 결과 요약
    print("\n학습 결과 요약:")
    print(f"  - 총 에포크: {NUM_EPOCHS}")
    print(f"  - 학습 샘플: {len(train_data):,}개")
    print(f"  - 검증 샘플: {len(dev_data):,}개")
    print(f"  - 최종 학습 손실: {train_loss:.4f}")
    print(f"  - 최종 검증 손실: {dev_loss:.4f}")


if __name__ == "__main__":
    main()


