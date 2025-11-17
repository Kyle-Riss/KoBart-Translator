# KoBART Multi-Task Learning Project

KoBART는 한국어에 특화된 BART (Bidirectional and Auto-Regressive Transformers) 모델입니다.

이 프로젝트는 **하나의 공유 인코더**와 **4개의 태스크별 디코더 헤드**를 가진 멀티태스크 학습 아키텍처를 구현합니다.

## 🎯 프로젝트 구조

```
입력 → Shared Encoder → 4개의 Decoder Heads
                         ├── Style Transfer
                         ├── Dialogue Summarization  
                         ├── Role-based Generation
                         └── QA Answer Generation
```

## 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install torch transformers sentencepiece
```

## 사용 방법

### 1. 기본 KoBART 모델

#### 기본 모델 로드 및 테스트

```bash
python3 quick_start.py        # 빠른 시작
python3 example_simple.py     # 상세 예제
python3 verify_installation.py # 설치 검증
```

### 2. Multi-Task KoBART 모델

#### 모델 테스트

```bash
python3 multi_task_kobart.py
```

#### 학습 시작

```bash
python3 train_multi_task.py
```

이 스크립트는 다음 작업을 수행합니다:
- 공유 인코더 로드
- 4개의 태스크별 디코더 생성
- 샘플 데이터로 학습

### Python 코드에서 직접 사용

```python
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')

# 텍스트 생성
text = "KoBART는 한국어에 특화된 BART 모델입니다."
inputs = tokenizer(text, return_tensors="pt")
output_ids = model.generate(inputs['input_ids'], max_length=50)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output)
```

## 주요 기능

### 기본 KoBART
- **요약 생성**: 긴 텍스트를 요약
- **텍스트 생성**: 주어진 프롬프트로부터 텍스트 생성
- **문장 변환**: 문장을 다른 형태로 변환

### Multi-Task KoBART (4개의 전문 디코더)
1. **Style Transfer**: 구어체 ↔ 격식체 변환
2. **Dialogue Summarization**: 대화 내용 요약
3. **Role-conditioned Generation**: 역할 기반 응답 생성 (선생님, 친구 등)
4. **QA Answer Generation**: 질문에 대한 답변 생성

## 모델 정보

- **모델명**: gogamza/kobart-base-v1
- **기반**: BART (Facebook AI)
- **언어**: 한국어
- **태스크**: 요약, 생성, 변환 등

## 시스템 요구사항

- Python 3.8 이상
- PyTorch 2.0 이상
- 최소 8GB RAM 권장
- GPU 사용 시 더 빠른 처리 가능 (선택사항)

## 📚 문서

- **MULTI_TASK_GUIDE.md**: 멀티태스크 사용 가이드
- **ARCHITECTURE.md**: 아키텍처 상세 설명
- **USAGE_GUIDE.md**: 기본 사용법
- **시작하기.md**: 빠른 시작 가이드 (한글)

## 📊 모델 정보

### 기본 KoBART
- 파라미터: ~124M

### Multi-Task KoBART
- 공유 인코더: ~66M 파라미터
- 4개 디코더: 각 ~103M 파라미터
- 총 파라미터: ~481M

## 참고 자료

- [Hugging Face Model Hub](https://huggingface.co/gogamza/kobart-base-v1)
- [BART 논문](https://arxiv.org/abs/1910.13461)


