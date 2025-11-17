# Multi-Task KoBART 아키텍처 가이드

## 🏗️ 아키텍처 구조

```
                    입력 문장
                        ↓
                   [Tokenizer]
                        ↓
                   Input IDs
                        ↓
        ╔═══════════════════════════════╗
        ║   Shared Encoder (KoBART)    ║  ← 모든 태스크가 공유
        ║   - 12 layers                 ║
        ║   - 768 hidden size           ║
        ╚═══════════════════════════════╝
                        ↓
                Context Vector
                        ↓
        ┌───────────────┬───────────────┐
        │               │               │
        ↓               ↓               ↓
    ┌────────┐      ┌────────┐     ┌────────┐     ┌────────┐
    │Decoder │      │Decoder │     │Decoder │     │Decoder │
    │Head 1  │      │Head 2  │     │Head 3  │     │Head 4  │
    └────────┘      └────────┘     └────────┘     └────────┘
        ↓               ↓               ↓             ↓
  Style          Dialogue        Role-based       QA Answer
  Transfer       Summary         Response         Generation
```

## 📊 구성 요소

### 1. Shared Encoder (공유 인코더)
- **역할**: 입력 문장을 컨텍스트 벡터로 인코딩
- **특징**: 
  - 모든 4개 태스크에서 공유
  - 멀티태스크 학습을 통해 범용적 표현 학습
  - 약 6천만 개 파라미터
- **장점**:
  - 메모리 효율적
  - 태스크 간 지식 공유
  - 전이 학습 효과

### 2. Task-Specific Decoders (태스크별 디코더)

#### Head 1: Style Transfer (스타일 변환)
- **목적**: 문장의 스타일 변환 (구어체 ↔ 격식체)
- **예시**:
  - 입력: "이거 좀 도와주세요"
  - 출력: "이것을 도와주시겠습니까?"

#### Head 2: Dialogue Summarization (대화 요약)
- **목적**: 대화 내용을 간결하게 요약
- **예시**:
  - 입력: "A: 내일 회의 몇 시에요? B: 오후 2시입니다."
  - 출력: "내일 회의는 오후 2시입니다."

#### Head 3: Role-conditioned Generation (역할 기반 응답)
- **목적**: 특정 역할에 맞는 응답 생성
- **예시**:
  - 입력: "[선생님] 파이썬이란?"
  - 출력: "파이썬은 배우기 쉬운 프로그래밍 언어입니다."

#### Head 4: QA Answer Generation (QA 답변 생성)
- **목적**: 질문에 대한 답변 생성
- **예시**:
  - 입력: "질문: 서울의 인구는?"
  - 출력: "서울의 인구는 약 1천만 명입니다."

## 🚀 사용 방법

### 기본 사용법

```python
from multi_task_kobart import MultiTaskKoBART
from transformers import PreTrainedTokenizerFast
import torch

# 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
model = MultiTaskKoBART()

# 입력 준비
text = "이거 좀 도와주세요."
inputs = tokenizer(text, return_tensors="pt")

# 태스크별 생성
output = model.generate(
    input_ids=inputs['input_ids'],
    task='style_transfer',  # 태스크 선택
    max_length=50
)

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
```

### 태스크별 사용 예제

#### 1. Style Transfer
```python
text = "빨리 와"
output = model.generate(
    input_ids=tokenizer(text, return_tensors="pt")['input_ids'],
    task='style_transfer',
    max_length=50
)
# Expected: "빠른 시일 내에 방문해 주시기 바랍니다."
```

#### 2. Dialogue Summarization
```python
dialogue = "A: 점심 뭐 먹을까요? B: 한식이 좋겠어요. A: 좋아요."
output = model.generate(
    input_ids=tokenizer(dialogue, return_tensors="pt")['input_ids'],
    task='dialogue_summarization',
    max_length=50
)
# Expected: "점심으로 한식을 먹기로 했습니다."
```

#### 3. Role-conditioned Generation
```python
prompt = "[친구] 주말에 뭐해?"
output = model.generate(
    input_ids=tokenizer(prompt, return_tensors="pt")['input_ids'],
    task='role_generation',
    max_length=50
)
# Expected: "특별한 계획은 없어. 너는?"
```

#### 4. QA Answer Generation
```python
question = "질문: 인공지능의 장점은?"
output = model.generate(
    input_ids=tokenizer(question, return_tensors="pt")['input_ids'],
    task='qa_generation',
    max_length=50
)
# Expected: "인공지능은 대량의 데이터를 빠르게 처리할 수 있습니다."
```

## 🎓 학습 방법

### 1. 전체 학습 (End-to-End)
```python
from train_multi_task import main

# 전체 파라미터 학습
main()
```

### 2. 인코더 고정 학습
```python
# 인코더 고정, 디코더만 학습
model = MultiTaskKoBART()
model.freeze_encoder()

# 이후 학습 진행
```

### 3. 태스크별 학습
```python
# 특정 태스크만 학습
task = 'style_transfer'
optimizer = torch.optim.AdamW(
    model.get_decoder_parameters(task),
    lr=5e-5
)
```

## 📈 학습 전략

### 1. 순차 학습 (Sequential Training)
```
Epoch 1-10: Task 1 학습
Epoch 11-20: Task 2 학습
Epoch 21-30: Task 3 학습
Epoch 31-40: Task 4 학습
```

### 2. 동시 학습 (Simultaneous Training)
```
각 배치마다 4개 태스크를 번갈아가며 학습
- 태스크 균형 유지 중요
- 데이터 샘플링 전략 필요
```

### 3. 2단계 학습 (Two-Stage Training)
```
Stage 1: 인코더 고정, 디코더만 학습 (빠름)
Stage 2: 전체 fine-tuning (정확도 향상)
```

## 💾 데이터 준비

### 데이터 포맷
```python
train_data = [
    {
        'task': 'style_transfer',
        'input': '입력 텍스트',
        'target': '타겟 텍스트'
    },
    # ... 더 많은 데이터
]
```

### 권장 데이터 크기
- **최소**: 태스크당 1,000개
- **권장**: 태스크당 10,000개 이상
- **최적**: 태스크당 100,000개 이상

### 데이터 균형
- 4개 태스크의 데이터 비율을 비슷하게 유지
- 불균형 시 샘플링 가중치 조정

## ⚙️ 고급 설정

### 1. 태스크별 가중치
```python
task_weights = {
    'style_transfer': 1.0,
    'dialogue_summarization': 1.5,
    'role_generation': 1.2,
    'qa_generation': 1.0
}

# Loss에 가중치 적용
weighted_loss = loss * task_weights[task]
```

### 2. 학습률 스케줄링
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### 3. Gradient Accumulation
```python
accumulation_steps = 4

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch, optimizer, device)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📊 성능 평가

### 태스크별 메트릭

| Task | Metric | 설명 |
|------|--------|------|
| Style Transfer | BLEU, Style Accuracy | 스타일 변환 정확도 |
| Dialogue Summary | ROUGE, BERTScore | 요약 품질 |
| Role Generation | Perplexity, Human Eval | 자연스러움 |
| QA Generation | F1, EM | 답변 정확도 |

## 🔧 문제 해결

### 1. 메모리 부족
- 배치 크기 줄이기
- Gradient checkpointing 사용
- 한 번에 한 태스크씩 학습

### 2. 특정 태스크 성능 저하
- 해당 태스크 데이터 증강
- 태스크별 학습률 조정
- 해당 디코더만 추가 학습

### 3. 태스크 간 간섭
- 인코더 고정 후 디코더만 학습
- 태스크별 순차 학습
- Adapter 레이어 추가

## 📚 참고 자료

- **논문**: "Multi-Task Learning with Deep Neural Networks"
- **KoBART**: https://huggingface.co/gogamza/kobart-base-v1
- **Transformers**: https://huggingface.co/docs/transformers

## 🎯 실전 활용

### 1. 챗봇 시스템
```
입력 → 의도 분류 → 적절한 디코더 선택 → 응답 생성
```

### 2. 문서 처리 시스템
```
문서 → 요약(Head 2) → 스타일 변환(Head 1) → 최종 문서
```

### 3. QA 시스템
```
질문 → QA 생성(Head 4) → 역할 기반 답변(Head 3)
```

---

**Created**: 2025-11-16
**Version**: 1.0
**License**: MIT

