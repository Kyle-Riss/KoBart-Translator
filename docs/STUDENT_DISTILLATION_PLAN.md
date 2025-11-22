# Ultra-Light Student Seq2Seq Plan

## 1. 목표
- 파라미터 1M~5M 범위의 초경량 encoder-decoder 구축
- 기존 multi-task 파이프라인과 호환 (가능하면 shared_text / qa 헤드 유지)
- teacher: `models/bart-ko-small` (fine-tuned MultiTaskKoBART)
- student: custom TinySeq2Seq (BART-like) + KD + pruning

## 2. 학생 모델 설계 초안
| 구성 | 값 (초기안) |
|------|-------------|
| vocab | 8k SentencePiece (재학습) |
| embedding share | encoder/decoder 완전 공유 |
| encoder_layers | 2 |
| decoder_layers | 2 |
| d_model | 128 |
| ffn_dim | 256 |
| num_heads | 1 |
| max_position_embeddings | 256 |
| dropout | 0.1 |

- `shared_text` / `qa_generation` 두 디코더를 유지하되, 각 모듈의 레이어 수 동일.
- LM 헤드도 공유 임베딩을 재활용하여 추가 파라미터 최소화.

## 3. 파이프라인 단계
1. **Teacher 준비**: 기존 MultiTaskKoBART(코스모) 체크포인트 사용, 필요 시 평가 모드.
2. **Student 초기화**:
   - 새로운 `BartConfig`로 TinySeq2Seq 인스턴스 생성.
   - SentencePiece 8k 모델을 `scripts/data/train_spm.py`(추가 예정)로 학습 후 tokenizer 교체.
3. **KD + Pruning 학습**:
   - loss = α * CE(student_logits, teacher_soft) + β * CE(student_logits, labels) + γ * MSE(hidden_states_enc/dec)
   - teacher attention head 중요도 기반 pruning으로 student 초기 가중치 세팅(옵션).
4. **Fine-tuning**:
   - KD 수렴 후, 실제 라벨 기반 CE로 몇 epoch 마무리.
5. **평가/로그**:
   - 파라미터 수, 모델 크기(MB), 추론 속도, 태스크별 ROUGE/F1 등 기록.

## 4. 구현 계획
- `kobart_translator/tiny_student.py`: TinySeq2Seq 클래스 + shared_text/qa 헤드.
- `scripts/training/train_student.py`: teacher/student 동시 로드, KD 루프.
- `scripts/data/train_spm.py`: 8k SentencePiece tokenizer 재학습 스크립트.

## 6. 현재 산출물
- `kobart_translator/tiny_student.py`: TinyStudentConfig + multi-head TinySeq2Seq PyTorch 구현.
- `scripts/data/train_spm.py`: 8k 토크나이저 학습 CLI (`--sources`, `--output_dir` 등).
- `scripts/training/train_student.py`: KD + hidden MSE + teacher 로깅이 포함된 학습 루프 초안.

## 5. 다음 작업
1. TinySeq2Seq config/클래스 초안 작성
2. SP tokenizer 재학습 스크립트 초안
3. KD 학습 스크립트 골격 작성

