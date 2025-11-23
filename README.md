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

## 📁 폴더 구성

- `kobart_translator/`: MultiTaskKoBART 및 데이터 로더 등 핵심 모듈
- `scripts/basic|demos|training|data/`: 실행·데모·학습·데이터 유틸 스크립트
- `tests/`: 회귀 및 유닛 테스트
- `docs/`: 아키텍처 및 사용 가이드 문서
- `data/`: 태스크별 데이터셋 (아래 참조)
- `logs/`: 학습/실행 로그

### 데이터셋 디렉터리

| Task | Path | 비고 |
| --- | --- | --- |
| Style Transfer | `data/style_transfer/korean_smile_style_dataset/` | Smilegate submodule |
| Dialogue Summarization | `data/dialogue_summarization/aihub_dialogue_summary/` | AI Hub 원본 ZIP |
| Role-conditioned Generation | `data/role_generation/aihub_dialogue_role_dataset/` | AI Hub 응급/오피스 대화 ZIP |
| QA Answer Generation | `data/qa/korquad/` | `scripts/data/download_korquad.py`로 KorQuAD1 저장, 2.0은 수동 추가 |

```bash
# KorQuAD1 train/dev JSON 자동 추출
python scripts/data/download_korquad.py --output data/qa/korquad

# 태스크별 JSONL 생성 (style/dialogue/role/qa)
python scripts/data/prepare_multitask_dataset.py --output-dir data/processed
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
python3 scripts/basic/quick_start.py           # 빠른 시작
python3 scripts/basic/example_simple.py        # 상세 예제
python3 scripts/basic/verify_installation.py   # 설치 검증
python3 scripts/basic/load_kobart.py           # 대화형 모드 포함
python3 scripts/demos/interactive_demo.py      # 실시간 데모
```

### 2. Multi-Task KoBART 모델

#### 모델 테스트

```bash
python3 -m kobart_translator.multi_task
```

#### 학습 시작

```bash
python3 scripts/training/train_multi_task.py
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

- **docs/MULTI_TASK_GUIDE.md**: 멀티태스크 사용 가이드
- **docs/ARCHITECTURE.md**: 아키텍처 상세 설명
- **docs/USAGE_GUIDE.md**: 기본 사용법

## 📊 모델 정보

### 기본 KoBART
- 파라미터: ~124M

### Multi-Task KoBART
- 공유 인코더: ~66M 파라미터
- 4개 디코더: 각 ~103M 파라미터
- 총 파라미터: ~481M

### Tiny Student Model (Knowledge Distillation)
- 초경량 모델: ~1M~5M 파라미터
- Teacher 모델로부터 지식 증류
- 자세한 내용은 `docs/STUDENT_DISTILLATION_PLAN.md` 참조

## 🚀 최근 개선사항

### 학습 개선
- **정규화**: Weight decay (기본값 0.01) 추가로 과적합 방지
- **Early Stopping**: 검증 손실이 개선되지 않으면 자동으로 학습 중단
- **Learning Rate Scheduler**: ReduceLROnPlateau 또는 CosineAnnealingLR 지원
- **Mixed Precision Training**: AMP 지원으로 메모리 효율성 향상
- **Gradient Checkpointing**: 메모리 사용량 감소 옵션

### 시각화 도구
- 학습 곡선 시각화: `scripts/utils/plot_training.py`
- 태스크별 손실 추적 및 비교
- TensorBoard 로깅 지원

### Knowledge Distillation
- Tiny Student 모델 구현 (`kobart_translator/tiny_student.py`)
- Teacher-Student 학습 스크립트 (`scripts/training/train_student.py`)
- 8K SentencePiece 토크나이저 재학습 지원

## 참고 자료

- [Hugging Face Model Hub](https://huggingface.co/gogamza/kobart-base-v1)
- [BART 논문](https://arxiv.org/abs/1910.13461)


