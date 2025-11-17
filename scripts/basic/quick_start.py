"""
KoBART 빠른 시작 가이드
"""

from transformers import pipeline

def main():
    print("="*60)
    print("KoBART 빠른 시작 예제")
    print("="*60)
    
    print("\n모델 로딩 중... (첫 실행시 다운로드가 필요합니다)")
    
    # Pipeline을 사용한 간단한 방법
    # 주의: KoBART는 요약 작업에 최적화되어 있습니다
    from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
    
    print("✓ 모델 로드 완료\n")
    
    # 예제 텍스트들
    examples = [
        "인공지능은 컴퓨터 시스템이 인간의 지능을 모방하는 기술입니다.",
        "서울은 대한민국의 수도이자 최대 도시입니다.",
        "파이썬은 배우기 쉽고 강력한 프로그래밍 언어입니다."
    ]
    
    print("테스트 예제:\n")
    
    for i, text in enumerate(examples, 1):
        print(f"[{i}] 입력: {text}")
        
        # 토큰화
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        # 생성
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            num_beams=3,
            early_stopping=True
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    출력: {result}\n")
    
    print("="*60)
    print("\n사용 가능한 스크립트:")
    print("  - scripts/basic/example_simple.py : 상세한 예제")
    print("  - scripts/basic/load_kobart.py : 대화형 모드 포함")
    print("  - scripts/demos/interactive_demo.py : 실시간 데모")
    print("  - scripts/basic/quick_start.py : 빠른 시작 (현재 파일)")
    print("\n실행 방법: python3 <파일명>")


if __name__ == "__main__":
    main()


