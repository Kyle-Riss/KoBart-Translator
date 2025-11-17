"""
KoBART 모델 로드 및 기본 사용 예제
"""

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import torch

def load_kobart_model():
    """KoBART 모델과 토크나이저를 로드합니다."""
    print("KoBART 모델을 로드하는 중...")
    
    # 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    print("✓ 토크나이저 로드 완료")
    
    # 모델 로드
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
    print("✓ 모델 로드 완료")
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✓ 디바이스: {device}")
    
    return model, tokenizer, device


def generate_text(model, tokenizer, device, input_text, max_length=50):
    """입력 텍스트로부터 요약을 생성합니다."""
    # 입력을 토큰화
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 텍스트 생성
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    # 결과 디코딩
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def main():
    """메인 함수"""
    # 모델 로드
    model, tokenizer, device = load_kobart_model()
    
    # 테스트 예제
    print("\n" + "="*50)
    print("KoBART 테스트 예제")
    print("="*50)
    
    test_texts = [
        "KoBART는 한국어에 특화된 BART 모델입니다. 다양한 자연어 처리 작업에 활용할 수 있습니다.",
        "인공지능 기술의 발전으로 자연어 처리 분야에서 많은 진전이 있었습니다. 특히 트랜스포머 기반 모델들이 좋은 성능을 보이고 있습니다.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[예제 {i}]")
        print(f"입력: {text}")
        
        result = generate_text(model, tokenizer, device, text)
        print(f"출력: {result}")
    
    print("\n" + "="*50)
    print("모델 로드 및 테스트 완료!")
    print("="*50)
    
    # 대화형 모드
    print("\n직접 텍스트를 입력해보세요 (종료하려면 'quit' 입력):")
    while True:
        user_input = input("\n입력: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료']:
            break
        if user_input:
            result = generate_text(model, tokenizer, device, user_input)
            print(f"출력: {result}")


if __name__ == "__main__":
    main()


