# NLP-Detection

한국어 자연어 처리(NLP) 기반 탐지 및 분석 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 한국어 텍스트 분석을 위한 다양한 도구와 모델을 통합하여, 텍스트 탐지, 분류, 감정 분석 등의 작업을 수행합니다.

## 서브모듈 구성

### 1. KoBART (Korean BART)

SKT-AI에서 개발한 한국어 BART 모델을 참조합니다.

- **Repository**: [https://github.com/SKT-AI/KoBART](https://github.com/SKT-AI/KoBART)
- **License**: Modified MIT License
- **Model**: 124M parameters, encoder-decoder 구조
- **Training Data**: 40GB+ 한국어 텍스트 (위키백과, 뉴스, 책, 모두의 말뭉치 등)

**주요 기능:**
- 텍스트 요약 (Summarization)
- 텍스트 분류 (Classification)
- 질의응답 (Question Answering)
- 텍스트 생성 (Text Generation)

### 2. Korean Smile Style Dataset

Smilegate AI에서 제공하는 한국어 스타일 분석 데이터셋을 참조합니다.

- **Repository**: [https://github.com/smilegate-ai/korean_smile_style_dataset](https://github.com/smilegate-ai/korean_smile_style_dataset)
- **License**: Apache License 2.0
- **Purpose**: 한국어 텍스트의 스타일 및 감정 분석

**주요 기능:**
- 한국어 텍스트 스타일 분석
- 감정 분류 데이터셋
- 텍스트 품질 평가

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

### 2. 필요한 패키지 설치

```bash
# KoBART 패키지 설치
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart

# 추가 의존성 설치
pip install transformers torch datasets
```

## 사용 예시

### KoBART 사용

```python
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
from transformers import BartModel

# 토크나이저 사용
kobart_tokenizer = get_kobart_tokenizer()
tokens = kobart_tokenizer.tokenize("안녕하세요. 한국어 BART 입니다.")
print(tokens)

# 모델 사용
model = BartModel.from_pretrained(get_pytorch_kobart_model())
inputs = kobart_tokenizer(['안녕하세요.'], return_tensors='pt')
output = model(inputs['input_ids'])
```

### Korean Smile Style Dataset 사용

```python
# 데이터셋 로드 (예시)
import pandas as pd
import json

# 데이터셋 파일 읽기
with open('korean_smile_style_dataset/data/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"훈련 데이터 개수: {len(train_data)}")
```

### 통합 분석 예시

```python
# KoBART와 Smile Style Dataset을 함께 사용한 텍스트 분석
def analyze_korean_text(text):
    # KoBART로 텍스트 인코딩
    tokens = kobart_tokenizer.tokenize(text)
    
    # 스타일 분석 (데이터셋 기반)
    # 여기에 분석 로직 구현
    
    return {
        'tokens': tokens,
        'style_analysis': '분석 결과'
    }
```

## 프로젝트 구조

```
NLP-Detection/
├── KoBART/                          # KoBART 서브모듈
│   ├── kobart/
│   ├── examples/
│   └── ...
├── korean_smile_style_dataset/       # Korean Smile Style Dataset 서브모듈
│   ├── data/
│   ├── scripts/
│   └── ...
├── .gitmodules                      # 서브모듈 설정
└── README.md                        # 프로젝트 설명
```

## 라이센스

- **KoBART**: Modified MIT License (자세한 내용은 `KoBART/LICENSE` 참조)
- **Korean Smile Style Dataset**: Apache License 2.0 (자세한 내용은 `korean_smile_style_dataset/LICENSE` 참조)
- **본 프로젝트**: MIT License

## 참고 자료

- [KoBART GitHub Repository](https://github.com/SKT-AI/KoBART)
- [Korean Smile Style Dataset](https://github.com/smilegate-ai/korean_smile_style_dataset)
- [KoBART 논문 - BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)

## 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 만듭니다 (`git checkout -b feature/new-feature`)
3. 변경사항을 커밋합니다 (`git commit -am 'Add new feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/new-feature`)
5. Pull Request를 생성합니다

---

**Note**: 이 프로젝트는 한국어 NLP 연구를 위해 KoBART와 Korean Smile Style Dataset을 통합한 것입니다. 각 서브모듈의 라이센스를 준수하여 사용하시기 바랍니다.

