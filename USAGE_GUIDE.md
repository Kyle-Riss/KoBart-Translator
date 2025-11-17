# KoBART 사용 가이드

## 📦 설치 완료!

KoBART 모델이 성공적으로 설치되었습니다!

## 🚀 빠른 시작

### 1. 가장 간단한 방법

```bash
python3 quick_start.py
```

### 2. 상세한 예제

```bash
python3 example_simple.py
```

### 3. 대화형 모드

```bash
python3 load_kobart.py
```

## 💡 Python 코드에서 사용하기

### 기본 사용법

```python
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')

# 텍스트 생성
text = "여기에 입력 텍스트를 넣으세요."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
```

### 고급 옵션

```python
# 더 나은 품질의 생성을 위한 파라미터 조정
outputs = model.generate(
    inputs['input_ids'],
    max_length=100,          # 최대 길이
    min_length=10,           # 최소 길이
    num_beams=5,             # Beam search 크기
    length_penalty=1.2,      # 길이 페널티
    early_stopping=True,     # 조기 종료
    no_repeat_ngram_size=3,  # 반복 방지
    temperature=0.8,         # 생성 다양성
    top_k=50,                # Top-k 샘플링
    top_p=0.95              # Top-p (nucleus) 샘플링
)
```

## 🔧 주요 기능

### 1. 텍스트 요약
KoBART는 긴 텍스트를 짧게 요약하는 데 특화되어 있습니다.

### 2. 텍스트 생성
주어진 문맥을 바탕으로 새로운 텍스트를 생성할 수 있습니다.

### 3. 문장 변환
문장을 다른 형태로 변환하는 작업에 사용할 수 있습니다.

## ⚙️ 파라미터 설명

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| max_length | 생성할 최대 토큰 수 | 50-200 |
| num_beams | Beam search 크기 (클수록 품질↑, 속도↓) | 3-5 |
| length_penalty | 길이 페널티 (>1: 긴 문장 선호) | 1.0-2.0 |
| no_repeat_ngram_size | n-gram 반복 방지 크기 | 2-3 |
| temperature | 생성 다양성 (높을수록 다양함) | 0.7-1.0 |

## 📊 모델 정보

- **모델명**: gogamza/kobart-base-v1
- **기반 구조**: BART (Facebook AI)
- **언어**: 한국어
- **훈련 데이터**: 한국어 코퍼스
- **용도**: 요약, 생성, 변환 작업

## 🎯 성능 최적화 팁

### 1. GPU 사용 (가능한 경우)
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### 2. 배치 처리
```python
texts = ["텍스트1", "텍스트2", "텍스트3"]
inputs = tokenizer(texts, return_tensors="pt", padding=True)
```

### 3. 모델 양자화 (메모리 절약)
```python
# 4-bit 양자화 예제 (별도 라이브러리 필요)
# pip install bitsandbytes
```

## 🐛 문제 해결

### 1. 메모리 부족
- `max_length` 값을 줄이기
- 배치 크기 줄이기
- 모델 양자화 사용

### 2. 느린 생성 속도
- `num_beams` 값을 줄이기
- GPU 사용하기
- 길이 제한 설정

### 3. 반복적인 출력
- `no_repeat_ngram_size` 증가
- `temperature` 조정
- `top_k`, `top_p` 파라미터 사용

## 📚 추가 학습 자료

- [Hugging Face 문서](https://huggingface.co/docs/transformers)
- [BART 논문](https://arxiv.org/abs/1910.13461)
- [KoBART 모델 페이지](https://huggingface.co/gogamza/kobart-base-v1)

## ⚠️ 주의사항

1. **첫 실행**: 모델 다운로드로 인해 시간이 걸릴 수 있습니다 (~500MB)
2. **메모리**: 최소 8GB RAM 권장
3. **출력 품질**: 기본 모델이므로 특정 작업에는 fine-tuning이 필요할 수 있습니다

## 📞 지원

문제가 발생하면:
1. 에러 메시지 확인
2. 파이썬/패키지 버전 확인
3. GitHub Issues 검색

---

**설치 날짜**: 2025-11-16
**Python 버전**: 3.13+
**주요 의존성**: PyTorch, Transformers, SentencePiece


