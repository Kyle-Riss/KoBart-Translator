# Multi-Task KoBART 아키텍처 상세 설명

## 🎯 프로젝트 개요

이 프로젝트는 **하나의 공유 인코더**와 **4개의 태스크별 디코더**를 가진 멀티태스크 학습 아키텍처를 구현합니다.

## 📐 아키텍처 다이어그램

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Text                            │
│                  "이거 좀 도와주세요"                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   Tokenizer      │
        │   (PreTrained)   │
        └─────────┬─────────┘
                  │
                  ▼ input_ids, attention_mask
        ┌─────────────────────────────┐
        │                             │
        │   SHARED ENCODER            │
        │   (KoBART Encoder)          │
        │                             │
        │   • 12 Transformer Layers   │
        │   • 768 Hidden Dim          │
        │   • 12 Attention Heads      │
        │   • ~66M Parameters         │
        │                             │
        └─────────┬───────────────────┘
                  │
                  ▼ encoder_hidden_states (Context Vector)
                  │
    ┌─────────────┼─────────────┬─────────────┐
    │             │             │             │
    ▼             ▼             ▼             ▼
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│Decoder │   │Decoder │   │Decoder │   │Decoder │
│Head 1  │   │Head 2  │   │Head 3  │   │Head 4  │
│        │   │        │   │        │   │        │
│Style   │   │Dialogue│   │ Role   │   │  QA    │
│Transfer│   │Summary │   │ Gen    │   │  Gen   │
│        │   │        │   │        │   │        │
│~103M   │   │~103M   │   │~103M   │   │~103M   │
│params  │   │params  │   │params  │   │params  │
└────┬───┘   └────┬───┘   └────┬───┘   └────┬───┘
     │            │            │            │
     ▼            ▼            ▼            ▼
  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
  │LM   │     │LM   │     │LM   │     │LM   │
  │Head │     │Head │     │Head │     │Head │
  └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
     │            │            │            │
     ▼            ▼            ▼            ▼
  Output1      Output2      Output3      Output4
```

## 🔍 컴포넌트 상세

### 1. Shared Encoder (공유 인코더)

**위치**: `model.shared_encoder`

**구조**:
```python
BartEncoder(
  (embed_tokens): Embedding(30000, 768)
  (embed_positions): LearnedPositionalEmbedding(1024, 768)
  (layers): ModuleList(
    (0-11): 12 x BartEncoderLayer(
      (self_attn): BartAttention(...)
      (self_attn_layer_norm): LayerNorm(...)
      (fc1): Linear(768, 3072)
      (fc2): Linear(3072, 768)
      (final_layer_norm): LayerNorm(...)
    )
  )
  (layernorm_embedding): LayerNorm(...)
)
```

**파라미터**:
- Embedding: 30,000 (vocab) × 768 = 23,040,000
- Positional: 1,024 × 768 = 786,432
- Transformer Layers: ~42,000,000
- **총 약 66M 파라미터**

**역할**:
1. 입력 텍스트를 토큰으로 변환
2. 토큰을 임베딩 벡터로 변환
3. 12개 레이어를 통과하며 컨텍스트 인코딩
4. 최종 hidden states 출력 (context vector)

### 2. Decoder Groups (태스크별 디코더)

**위치 / 매핑**: 
- `model.decoders['shared_text']` → `style_transfer`, `dialogue_summarization`, `role_generation`
- `model.decoders['qa_generation']` → `qa_generation`

**구조** (각 디코더):
```python
BartDecoder(
  (embed_tokens): Embedding(30000, 768)
  (embed_positions): LearnedPositionalEmbedding(1024, 768)
  (layers): ModuleList(
    (0-11): 12 x BartDecoderLayer(
      (self_attn): BartAttention(...)
      (encoder_attn): BartAttention(...)  # Cross-attention
      (self_attn_layer_norm): LayerNorm(...)
      (encoder_attn_layer_norm): LayerNorm(...)
      (fc1): Linear(768, 3072)
      (fc2): Linear(3072, 768)
      (final_layer_norm): LayerNorm(...)
    )
  )
  (layernorm_embedding): LayerNorm(...)
)
```

**파라미터** (디코더 1개):
- 약 103M 파라미터
- **현재 구성:** 공유 Text 디코더 + QA 디코더 = 총 약 206M

**역할**:
1. 인코더의 context vector를 받음
2. Cross-attention으로 인코더 정보 활용
3. Self-attention으로 이전 토큰 참조
4. 태스크 그룹에 특화된 출력 생성 (style/summary/role은 하나의 디코더 공유)

### 3. Language Model Heads (LM 헤드)

**위치 / 매핑**:
- `model.lm_heads['shared_text']` → `style_transfer`, `dialogue_summarization`, `role_generation`
- `model.lm_heads['qa_generation']` → `qa_generation`

**구조**:
```python
Linear(768, 30000)  # hidden_size → vocab_size
```

**파라미터** (헤드 1개):
- 768 × 30,000 = 23,040,000
- **현재 구성:** 2개 헤드 → 총 약 46M 파라미터

**역할**:
1. 디코더 출력 (768차원)을 어휘 크기(30,000)로 변환
2. 각 토큰의 확률 분포 생성
3. 최종 토큰 예측

## 📊 전체 파라미터 통계

| 컴포넌트 | 파라미터 수 | 비율(대략) |
|---------|-----------|-----------|
| Shared Encoder | 66M | 24% |
| Shared Text Decoder | 103M | 37% |
| QA Decoder | 103M | 37% |
| LM Heads (2) | 23M × 2 | 12% |
| **총합** | **~295M** | **100%** |

## 🔄 데이터 흐름

### Forward Pass

```python
# 1. 입력 준비
text = "이거 좀 도와주세요"
tokens = tokenizer(text, return_tensors="pt")
# tokens.shape: [batch_size, seq_len]

# 2. 인코더 통과
encoder_output = model.shared_encoder(tokens['input_ids'])
# encoder_output.shape: [batch_size, seq_len, 768]

# 3. 태스크 선택
task = 'style_transfer'
decoder_key = model.task_to_decoder[task]
decoder = model.decoders[decoder_key]

# 4. 디코더 통과
decoder_output = decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_output
)
# decoder_output.shape: [batch_size, target_len, 768]

# 5. LM Head 통과
logits = model.lm_heads[decoder_key](decoder_output)
# logits.shape: [batch_size, target_len, 30000]

# 6. 토큰 예측
predicted_tokens = torch.argmax(logits, dim=-1)
# predicted_tokens.shape: [batch_size, target_len]

# 7. 디코딩
output_text = tokenizer.decode(predicted_tokens[0])
```

### Training Flow

```python
# 1. 데이터 로드
batch = {
    'task': 'style_transfer',
    'input': "이거 좀 도와주세요",
    'target': "이것을 도와주시겠습니까?"
}

# 2. Forward pass
outputs = model(
    input_ids=input_tokens,
    decoder_input_ids=target_tokens,
    task=batch['task']
)

# 3. Loss 계산
loss = CrossEntropyLoss(outputs['logits'], labels)

# 4. Backward pass
loss.backward()

# 5. 파라미터 업데이트
optimizer.step()
```

## 🎓 학습 전략

### 전략 1: Joint Training (동시 학습)

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        task = batch['task']  # 배치마다 태스크가 다름
        
        # Forward
        outputs = model(
            input_ids=batch['input_ids'],
            decoder_input_ids=batch['decoder_input_ids'],
            task=task
        )
        
        # Loss & Backward
        loss = compute_loss(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
```

**장점**:
- 인코더가 모든 태스크를 동시에 학습
- 태스크 간 지식 공유 효과
- 범용적인 표현 학습

**단점**:
- 태스크 간 간섭 가능
- 데이터 균형 중요

### 전략 2: Sequential Training (순차 학습)

```python
tasks = ['style_transfer', 'dialogue_summarization', 
         'role_generation', 'qa_generation']

for task in tasks:
    print(f"Training {task}...")
    
    # 해당 태스크 데이터만 사용
    task_dataloader = get_task_dataloader(task)
    
    for epoch in range(epochs_per_task):
        for batch in task_dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                decoder_input_ids=batch['decoder_input_ids'],
                task=task
            )
            
            loss = compute_loss(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
```

**장점**:
- 태스크별 집중 학습
- 간섭 최소화
- 구현 단순

**단점**:
- 학습 시간 길어짐
- 이전 태스크 망각 가능 (Catastrophic Forgetting)

### 전략 3: Two-Stage Training (2단계 학습)

```python
# Stage 1: 인코더 고정, 디코더만 학습
model.freeze_encoder()

for task in tasks:
    decoder_optimizer = AdamW(
        model.get_decoder_parameters(task),
        lr=5e-5
    )
    
    # 태스크별 디코더 학습
    train_decoder(model, task, decoder_optimizer)

# Stage 2: 전체 fine-tuning
model.unfreeze_encoder()
full_optimizer = AdamW(model.parameters(), lr=1e-5)

# 전체 모델 미세 조정
fine_tune_all(model, full_optimizer)
```

**장점**:
- 빠른 초기 학습
- 안정적인 수렴
- 리소스 효율적

**단점**:
- 2단계 관리 필요
- Stage 1에서 인코더 개선 불가

## 🔧 고급 기법

### 1. Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(...)
    loss = compute_loss(outputs, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(...)
        loss = compute_loss(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Task Sampling Strategy

```python
# 태스크별 샘플링 확률 조정
task_weights = {
    'style_transfer': 0.25,
    'dialogue_summarization': 0.35,  # 더 많이 샘플링
    'role_generation': 0.20,
    'qa_generation': 0.20
}

sampler = WeightedTaskSampler(dataset, task_weights)
dataloader = DataLoader(dataset, sampler=sampler)
```

## 💾 모델 저장 및 로드

### 전체 모델 저장

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'multi_task_kobart.pt')
```

### 태스크별 디코더만 저장

```python
task = 'style_transfer'
torch.save({
    'decoder_state_dict': model.decoders[task].state_dict(),
    'lm_head_state_dict': model.lm_heads[task].state_dict(),
}, f'{task}_decoder.pt')
```

### 로드

```python
checkpoint = torch.load('multi_task_kobart.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 🎯 실전 활용 예제

### 챗봇 시스템

```python
def chatbot_pipeline(user_input, user_role='friend'):
    # 1. 의도 분류
    intent = classify_intent(user_input)
    
    # 2. 태스크 매핑
    task_map = {
        'question': 'qa_generation',
        'chat': 'role_generation',
        'summarize': 'dialogue_summarization',
        'formalize': 'style_transfer'
    }
    task = task_map[intent]
    
    # 3. 역할 추가
    if task == 'role_generation':
        user_input = f"[{user_role}] {user_input}"
    
    # 4. 생성
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        task=task
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 성능 모니터링

```python
def evaluate_all_tasks(model, test_dataloaders):
    results = {}
    
    for task, dataloader in test_dataloaders.items():
        metrics = evaluate_task(model, dataloader, task)
        results[task] = metrics
        
        print(f"{task}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  BLEU: {metrics['bleu']:.4f}")
        print(f"  ROUGE: {metrics['rouge']:.4f}")
    
    return results
```

---

**작성일**: 2025-11-16
**버전**: 1.0
**저자**: Multi-Task KoBART Team

