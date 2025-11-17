"""
학습된 스타일 변환 모델 테스트
"""

import torch
from transformers import PreTrainedTokenizerFast
from kobart_translator import MultiTaskKoBART, StyleTransferDataLoader


def load_model(checkpoint_path: str, device):
    """학습된 모델 로드"""
    print("모델 로딩 중...")
    
    model = MultiTaskKoBART()
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ 모델 로드 완료")
    if 'epoch' in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    if 'dev_loss' in checkpoint:
        print(f"  - Dev Loss: {checkpoint['dev_loss']:.4f}")
    
    return model


def generate_text(model, tokenizer, input_text, source_style, target_style, device, max_length=128):
    """텍스트 생성"""
    # 입력 형식: [source→target] text
    formatted_input = f"[{source_style}→{target_style}] {input_text}"
    
    # 토큰화
    inputs = tokenizer(
        formatted_input,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            task='style_transfer',
            max_length=max_length
        )
    
    # 디코딩
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def test_samples(model, tokenizer, device):
    """샘플 테스트"""
    print("\n" + "="*60)
    print("샘플 테스트")
    print("="*60)
    
    test_cases = [
        # 반말 → 요체
        {
            'input': '응. 오늘 날씨 좋다.',
            'source': 'ban',
            'target': 'yo',
            'expected': '네. 오늘 날씨 좋아요.'
        },
        {
            'input': '뭐해? 심심한데.',
            'source': 'ban',
            'target': 'yo',
            'expected': '뭐해요? 심심한데요.'
        },
        
        # 반말 → 합쇼체
        {
            'input': '이거 좀 도와줘.',
            'source': 'ban',
            'target': 'sho',
            'expected': '이것을 좀 도와주십시오.'
        },
        {
            'input': '회의 시작했어.',
            'source': 'ban',
            'target': 'sho',
            'expected': '회의 시작했습니다.'
        },
        
        # 요체 → 반말
        {
            'input': '네. 알겠어요.',
            'source': 'yo',
            'target': 'ban',
            'expected': '응. 알겠어.'
        },
        
        # 요체 → 합쇼체
        {
            'input': '오늘 회의가 있어요.',
            'source': 'yo',
            'target': 'sho',
            'expected': '오늘 회의가 있습니다.'
        },
        
        # 합쇼체 → 반말
        {
            'input': '준비해주십시오.',
            'source': 'sho',
            'target': 'ban',
            'expected': '준비해줘.'
        },
        
        # 합쇼체 → 요체
        {
            'input': '확인해주시기 바랍니다.',
            'source': 'sho',
            'target': 'yo',
            'expected': '확인해주세요.'
        },
    ]
    
    style_names = {
        'ban': '반말',
        'yo': '요체',
        'sho': '합쇼체'
    }
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {style_names[test['source']]} → {style_names[test['target']]}")
        print(f"입력:   {test['input']}")
        print(f"기대값: {test['expected']}")
        
        result = generate_text(
            model, tokenizer,
            test['input'],
            test['source'],
            test['target'],
            device
        )
        print(f"출력:   {result}")
        
        # 간단한 평가
        if result.strip() == test['expected'].strip():
            print("✓ 정확히 일치!")
        elif test['expected'][:10] in result or result[:10] in test['expected']:
            print("△ 유사함")
        else:
            print("✗ 다름")


def test_with_real_data(model, tokenizer, device):
    """실제 테스트 데이터로 평가"""
    print("\n" + "="*60)
    print("실제 테스트 데이터 평가")
    print("="*60)
    
    # 데이터 로더
    data_loader = StyleTransferDataLoader()
    test_data = data_loader.get_test_data()
    
    print(f"\n테스트 데이터: {len(test_data)}개")
    
    # 랜덤하게 10개 샘플 테스트
    import random
    samples = random.sample(test_data, min(10, len(test_data)))
    
    style_names = {
        'ban': '반말',
        'yo': '요체',
        'sho': '합쇼체'
    }
    
    for i, item in enumerate(samples, 1):
        print(f"\n테스트 {i}: {style_names[item['source_style']]} → {style_names[item['target_style']]}")
        print(f"입력:   {item['input']}")
        print(f"정답:   {item['target']}")
        
        result = generate_text(
            model, tokenizer,
            item['input'],
            item['source_style'],
            item['target_style'],
            device
        )
        print(f"출력:   {result}")


def interactive_mode(model, tokenizer, device):
    """대화형 모드"""
    print("\n" + "="*60)
    print("대화형 스타일 변환")
    print("="*60)
    
    style_names = {
        '1': ('ban', '반말'),
        '2': ('yo', '요체'),
        '3': ('sho', '합쇼체')
    }
    
    print("\n사용 가능한 스타일:")
    for key, (code, name) in style_names.items():
        print(f"  {key}. {name} ({code})")
    
    while True:
        print("\n" + "-"*60)
        
        # 소스 스타일 선택
        source = input("\n원본 스타일 (1/2/3, 종료: q): ").strip()
        if source.lower() == 'q':
            break
        
        if source not in style_names:
            print("잘못된 선택입니다.")
            continue
        
        # 타겟 스타일 선택
        target = input("변환할 스타일 (1/2/3): ").strip()
        if target not in style_names:
            print("잘못된 선택입니다.")
            continue
        
        source_code, source_name = style_names[source]
        target_code, target_name = style_names[target]
        
        if source_code == target_code:
            print("같은 스타일입니다.")
            continue
        
        # 텍스트 입력
        text = input(f"\n{source_name} 텍스트 입력: ").strip()
        if not text:
            continue
        
        print(f"\n변환 중... ({source_name} → {target_name})")
        result = generate_text(
            model, tokenizer,
            text,
            source_code,
            target_code,
            device
        )
        print(f"결과: {result}")


def main():
    """메인 함수"""
    print("="*60)
    print("스타일 변환 모델 테스트")
    print("="*60)
    print()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}\n")
    
    # 토크나이저 로드
    print("토크나이저 로딩 중...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    print("✓ 토크나이저 로드 완료\n")
    
    # 모델 로드
    checkpoint_path = "/Users/arka/Desktop/Ko-bart/checkpoints/best_model.pt"
    
    try:
        model = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("\n먼저 train_style_transfer.py를 실행하여 모델을 학습시켜주세요.")
        return
    
    # 메뉴
    while True:
        print("\n" + "="*60)
        print("테스트 메뉴")
        print("="*60)
        print("1. 샘플 테스트")
        print("2. 실제 데이터 테스트")
        print("3. 대화형 모드")
        print("0. 종료")
        print("="*60)
        
        choice = input("\n선택: ").strip()
        
        if choice == '0':
            print("\n종료합니다.")
            break
        elif choice == '1':
            test_samples(model, tokenizer, device)
        elif choice == '2':
            test_with_real_data(model, tokenizer, device)
        elif choice == '3':
            interactive_mode(model, tokenizer, device)
        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()


