# NLP-Detection

자연어 처리(NLP) 기반 탐지 프로젝트입니다.

## 프로젝트 구조

이 프로젝트는 한국어 NLP 모델을 활용한 탐지 시스템을 구축합니다.

### KoBART (Korean BART) 참조

이 프로젝트는 SKT-AI의 **KoBART**(Korean Bidirectional and Auto-Regressive Transformers)를 서브모듈로 참조합니다.

- **KoBART Repository**: [https://github.com/SKT-AI/KoBART](https://github.com/SKT-AI/KoBART)
- **License**: Modified MIT License
- **Model**: 124M parameters, encoder-decoder 구조
- **Training Data**: 40GB+ 한국어 텍스트 (위키백과, 뉴스, 책, 모두의 말뭉치 등)

KoBART는 다음과 같은 작업에 활용될 수 있습니다:
- 텍스트 요약 (Summarization)
- 텍스트 분류 (Classification)
- 질의응답 (Question Answering)
- 텍스트 생성 (Text Generation)

## 설치 방법

### 1. 레포지토리 클론 (서브모듈 포함)

```bash
# 서브모듈과 함께 클론
git clone --recursive https://github.com/Kyle-Riss/NLP-Detection.git

# 또는 이미 클론한 경우 서브모듈 초기화
git clone https://github.com/Kyle-Riss/NLP-Detection.git
cd NLP-Detection
git submodule init
git submodule update
```

### 2. KoBART 설치

```bash
# KoBART 패키지 설치
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```

## 사용 예시

### KoBART Tokenizer 사용

```python
from kobart import get_kobart_tokenizer

kobart_tokenizer = get_kobart_tokenizer()
tokens = kobart_tokenizer.tokenize("안녕하세요. 한국어 BART 입니다.")
print(tokens)
```

### KoBART Model 사용

```python
from transformers import BartModel
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

kobart_tokenizer = get_kobart_tokenizer()
model = BartModel.from_pretrained(get_pytorch_kobart_model())
inputs = kobart_tokenizer(['안녕하세요.'], return_tensors='pt')
output = model(inputs['input_ids'])
```

## 라이센스

- **KoBART**: Modified MIT License (자세한 내용은 `KoBART/LICENSE` 참조)
- 본 프로젝트: MIT License

## 참고 자료

- [KoBART GitHub Repository](https://github.com/SKT-AI/KoBART)
- [KoBART 논문 - BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)

---

**Note**: KoBART는 SKT-AI에서 개발한 한국어 특화 BART 모델입니다. 본 프로젝트에서는 이를 참조하여 한국어 NLP 작업을 수행합니다.

