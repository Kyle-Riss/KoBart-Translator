"""
Multi-Task KoBART 학습 예제
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from kobart_translator import MultiTaskKoBART
from transformers import PreTrainedTokenizerFast
from typing import Dict, List


class MultiTaskDataset(Dataset):
    """멀티태스크 학습용 데이터셋"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512
    ):
        """
        Args:
            data: 데이터 리스트 [{'task': task_name, 'input': text, 'target': text}, ...]
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 입력 토큰화
        inputs = self.tokenizer(
            item['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 타겟 토큰화
        targets = self.tokenizer(
            item['target'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'task': item['task'],
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0)
        }


def create_sample_dataset() -> List[Dict]:
    """샘플 학습 데이터 생성"""
    return [
        # Style Transfer 데이터
        {
            'task': 'style_transfer',
            'input': '이거 좀 도와주세요.',
            'target': '이것을 도와주시겠습니까?'
        },
        {
            'task': 'style_transfer',
            'input': '빨리 와.',
            'target': '빠른 시일 내에 방문해 주시기 바랍니다.'
        },
        
        # Dialogue Summarization 데이터
        {
            'task': 'dialogue_summarization',
            'input': 'A: 내일 회의 몇 시에요? B: 오후 2시입니다. A: 알겠습니다.',
            'target': '내일 회의는 오후 2시에 있습니다.'
        },
        {
            'task': 'dialogue_summarization',
            'input': 'A: 점심 뭐 먹을까요? B: 한식이 좋겠어요. A: 좋아요.',
            'target': '점심으로 한식을 먹기로 했습니다.'
        },
        
        # Role-conditioned Generation 데이터
        {
            'task': 'role_generation',
            'input': '[선생님] 파이썬이란 무엇인가요?',
            'target': '파이썬은 배우기 쉬운 프로그래밍 언어로, 다양한 분야에서 활용됩니다.'
        },
        {
            'task': 'role_generation',
            'input': '[친구] 주말에 뭐해?',
            'target': '특별한 계획은 없어. 너는?'
        },
        
        # QA Generation 데이터
        {
            'task': 'qa_generation',
            'input': '질문: 서울의 인구는 얼마나 되나요?',
            'target': '서울의 인구는 약 1천만 명입니다.'
        },
        {
            'task': 'qa_generation',
            'input': '질문: 인공지능의 장점은 무엇인가요?',
            'target': '인공지능은 대량의 데이터를 빠르게 처리하고 패턴을 학습할 수 있습니다.'
        }
    ]


def train_step(
    model: MultiTaskKoBART,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """한 스텝 학습"""
    model.train()
    
    # 데이터를 디바이스로 이동
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    task = batch['task'][0]  # 배치 내 모든 샘플이 같은 태스크라고 가정
    
    # Decoder input ids 생성 (labels를 오른쪽으로 시프트)
    decoder_input_ids = labels.clone()
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = model.config.decoder_start_token_id
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        task=task
    )
    
    # Loss 계산
    logits = outputs['logits']
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_epoch(
    model: MultiTaskKoBART,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
):
    """한 에포크 학습"""
    total_loss = 0
    task_losses = {'style_transfer': [], 'dialogue_summarization': [], 
                   'role_generation': [], 'qa_generation': []}
    
    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(model, batch, optimizer, device)
        total_loss += loss
        
        task = batch['task'][0]
        task_losses[task].append(loss)
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch} | Batch {batch_idx + 1} | Loss: {avg_loss:.4f}")
    
    # 태스크별 평균 손실 출력
    print(f"\n태스크별 평균 손실:")
    for task, losses in task_losses.items():
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"  {task}: {avg_loss:.4f}")
    
    return total_loss / len(dataloader)


def main():
    """학습 메인 함수"""
    print("="*60)
    print("Multi-Task KoBART 학습 예제")
    print("="*60)
    print()
    
    # 하이퍼파라미터
    BATCH_SIZE = 2
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}\n")
    
    # 토크나이저 및 모델 로드
    print("모델 로딩 중...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    model = MultiTaskKoBART()
    model.to(device)
    print("✓ 모델 로드 완료\n")
    
    # 샘플 데이터 생성
    print("데이터 준비 중...")
    train_data = create_sample_dataset()
    train_dataset = MultiTaskDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✓ 총 {len(train_data)}개 샘플\n")
    
    # Optimizer 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 학습
    print("학습 시작...\n")
    print("-"*60)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")
        avg_loss = train_epoch(model, train_dataloader, optimizer, device, epoch)
        print(f"\nEpoch {epoch} 평균 손실: {avg_loss:.4f}")
        print("-"*60)
    
    print("\n✓ 학습 완료!")
    
    # 모델 저장
    save_path = "/Users/arka/Desktop/Ko-bart/multi_task_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"\n모델 저장 완료: {save_path}")
    
    print("\n" + "="*60)
    print("학습 팁:")
    print("1. 실제 학습에는 더 많은 데이터가 필요합니다")
    print("2. 태스크별로 균형잡힌 데이터셋 구성이 중요합니다")
    print("3. 인코더를 고정하고 디코더만 학습할 수도 있습니다:")
    print("   model.freeze_encoder()")
    print("4. 태스크별로 별도의 learning rate 적용 가능")
    print("="*60)


if __name__ == "__main__":
    main()

