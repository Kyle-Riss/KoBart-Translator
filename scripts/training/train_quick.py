"""
빠른 테스트를 위한 소규모 학습 (데이터 1000개, 2 에포크)
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
    
    def __init__(self, data, tokenizer, max_length: int = 128):
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


def main():
    """빠른 테스트 학습"""
    print("="*60)
    print("빠른 테스트 학습 (소규모)")
    print("="*60)
    print()
    
    # 하이퍼파라미터
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 5
    MAX_LENGTH = 128
    MAX_SAMPLES = 70000  # 전체 데이터로 학습
    SAVE_DIR = "checkpoints"
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 디렉토리 생성
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 토크나이저 로드
    print("\n토크나이저 로딩 중...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    print("[OK] 토크나이저 로드 완료")
    
    # 모델 로드
    print("\n모델 로딩 중...")
    model = MultiTaskKoBART()
    model.to(device)
    print("[OK] 모델 로드 완료")
    
    # 데이터 로드
    print("\n데이터 로딩 중...")
    data_loader = StyleTransferDataLoader()
    
    # 학습 데이터 (전체 - 병렬 + OPUS)
    train_data = data_loader.get_train_data(
        use_parallel=True, 
        use_opus=True,
        opus_samples=100000,  # OPUS 스타일당 100,000개 (총 300,000개)
        parallel_samples=None  # 병렬 전체 사용
    )
    print(f"[OK] 전체 학습 데이터: {len(train_data):,}개")
    
    # 검증 데이터
    dev_data = data_loader.get_dev_data()
    dev_data = dev_data[:100]  # 100개만 사용
    print(f"[OK] 검증 데이터: {len(dev_data):,}개")
    
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
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"학습")
        
        for batch in progress_bar:
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Decoder input ids 생성
            decoder_input_ids = labels.clone()
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = model.config.decoder_start_token_id
            
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress bar 업데이트
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        train_loss = total_loss / num_batches
        
        # 검증
        print("검증 중...")
        model.eval()
        dev_total_loss = 0
        dev_num_batches = 0
        
        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                decoder_input_ids = labels.clone()
                decoder_input_ids[:, 1:] = labels[:, :-1]
                decoder_input_ids[:, 0] = model.config.decoder_start_token_id
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    task='style_transfer'
                )
                
                logits = outputs['logits']
                loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                dev_total_loss += loss.item()
                dev_num_batches += 1
        
        dev_loss = dev_total_loss / dev_num_batches
        
        print(f"\nEpoch {epoch} 결과:")
        print(f"  - Train Loss: {train_loss:.4f}")
        print(f"  - Dev Loss: {dev_loss:.4f}")
        
        # 모델 저장
        save_path = os.path.join(SAVE_DIR, f"quick_test_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'dev_loss': dev_loss,
        }, save_path)
        print(f"[OK] 모델 저장: {save_path}")
    
    print("\n" + "="*60)
    print("빠른 테스트 학습 완료!")
    print("="*60)
    print(f"\n다음 명령어로 테스트:")
    print(f"  python3 test_quick.py")


if __name__ == "__main__":
    main()

