"""
KoBART 간단한 사용 예제
"""

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import torch

def main():
    print("KoBART 모델 로딩 중...")
    
    # 토크나이저 및 모델 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✓ 모델 로드 완료 (디바이스: {device})")
    print("\n" + "="*60)
    
    # 테스트 텍스트
    test_text = """
    한국은 아시아 대륙 동쪽 끝에 위치한 반도 국가입니다. 
    수도는 서울이며, 인구는 약 5천만 명입니다. 
    한국은 빠른 경제 성장과 기술 발전으로 유명합니다.
    """
    
    print("원본 텍스트:")
    print(test_text.strip())
    print("\n" + "-"*60)
    
    # 텍스트 토큰화
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 요약 생성
    print("\n요약 생성 중...")
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=50,
            min_length=10,
            num_beams=5,
            length_penalty=1.2,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # 결과 디코딩
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print("\n생성된 요약:")
    print(summary)
    print("\n" + "="*60)
    print("\n✓ 완료!")
    
    # 모델 정보 출력
    print("\n모델 정보:")
    print(f"  - 모델명: gogamza/kobart-base-v1")
    print(f"  - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 학습 가능 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    main()


