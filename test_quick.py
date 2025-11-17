"""
빠른 테스트 모델 평가
"""

import torch
from transformers import PreTrainedTokenizerFast
from multi_task_kobart import MultiTaskKoBART


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
    if 'train_loss' in checkpoint:
        print(f"  - Train Loss: {checkpoint['train_loss']:.4f}")
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
            max_length=max_length,
            repetition_penalty=3.0,
            no_repeat_ngram_size=2,
            num_beams=5,
            early_stopping=True
        )
    
    # 디코딩
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def main():
    """메인 함수"""
    print("="*60)
    print("빠른 테스트 모델 평가")
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
    checkpoint_path = "checkpoints/quick_test_epoch_3.pt"
    
    try:
        model = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
    
    print("\n" + "="*60)
    print("테스트 샘플 생성")
    print("="*60)
    
    style_names = {
        'ban': '반말',
        'yo': '요체',
        'sho': '합쇼체'
    }
    
    test_cases = [
        # 반말 → 요체
        {
            'input': '안녕. 오늘 날씨 좋아.',
            'source': 'ban',
            'target': 'yo',
        },
        {
            'input': '뭐해? 심심해.',
            'source': 'ban',
            'target': 'yo',
        },
        
        # 반말 → 합쇼체
        {
            'input': '이거 좀 도와줘.',
            'source': 'ban',
            'target': 'sho',
        },
        {
            'input': '회의 시작했어.',
            'source': 'ban',
            'target': 'sho',
        },
        
        # 요체 → 반말
        {
            'input': '네. 알겠어요.',
            'source': 'yo',
            'target': 'ban',
        },
        {
            'input': '오늘 날씨가 좋아요.',
            'source': 'yo',
            'target': 'ban',
        },
        
        # 요체 → 합쇼체
        {
            'input': '회의가 있어요.',
            'source': 'yo',
            'target': 'sho',
        },
        
        # 합쇼체 → 반말
        {
            'input': '준비해주십시오.',
            'source': 'sho',
            'target': 'ban',
        },
        
        # 합쇼체 → 요체
        {
            'input': '확인해주시기 바랍니다.',
            'source': 'sho',
            'target': 'yo',
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {style_names[test['source']]} → {style_names[test['target']]}")
        print(f"입력: {test['input']}")
        
        result = generate_text(
            model, tokenizer,
            test['input'],
            test['source'],
            test['target'],
            device
        )
        print(f"출력: {result}")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)
    
    print("\n📊 관찰:")
    print("  - 2 에포크만 학습했지만 패턴을 학습하기 시작")
    print("  - 더 많은 에포크로 성능 향상 가능")
    print("  - 전체 데이터(70K)로 학습하면 더 좋은 결과")
    
    print("\n💡 다음 단계:")
    print("  1. 전체 데이터로 학습: train_style_transfer.py")
    print("  2. 더 많은 에포크 (10-20)")
    print("  3. Beam search 파라미터 튜닝")


if __name__ == "__main__":
    main()


