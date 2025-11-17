"""
학습된 Multi-Task KoBART 모델 테스트
"""

import torch
from kobart_translator import MultiTaskKoBART
from transformers import PreTrainedTokenizerFast


def load_trained_model(checkpoint_path: str):
    """학습된 모델 로드"""
    print("학습된 모델 로딩 중...")
    
    model = MultiTaskKoBART()
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ 학습된 모델 로드 완료")
    
    return model


def test_task(model, tokenizer, task, test_cases, device):
    """특정 태스크 테스트"""
    print(f"\n{'='*60}")
    print(f"[{task.upper().replace('_', ' ')}]")
    print('='*60)
    
    with torch.no_grad():
        for i, (input_text, expected) in enumerate(test_cases, 1):
            print(f"\n테스트 {i}:")
            print(f"입력: {input_text}")
            if expected:
                print(f"기대값: {expected}")
            
            # 토큰화
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 생성
            try:
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    task=task,
                    max_length=100
                )
                
                # 디코딩
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"출력: {result}")
                
            except Exception as e:
                print(f"생성 중 오류: {e}")


def main():
    """메인 함수"""
    print("="*60)
    print("학습된 Multi-Task KoBART 모델 테스트")
    print("="*60)
    print()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}\n")
    
    # 토크나이저 로드
    print("토크나이저 로딩 중...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    print("✓ 토크나이저 로드 완료\n")
    
    # 학습된 모델 로드
    checkpoint_path = "/Users/arka/Desktop/Ko-bart/multi_task_model.pt"
    model = load_trained_model(checkpoint_path)
    model.to(device)
    
    print(f"\n모델 정보:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 전체 파라미터: {total_params:,}")
    print(f"  - 디바이스: {device}")
    
    # 태스크별 테스트 케이스
    test_data = {
        'style_transfer': [
            ("이거 좀 도와주세요.", "이것을 도와주시겠습니까?"),
            ("빨리 와.", "빠른 시일 내에 방문해 주시기 바랍니다."),
            ("뭐해?", "무엇을 하고 계십니까?"),
            ("고마워.", "감사합니다."),
        ],
        
        'dialogue_summarization': [
            ("A: 내일 회의 몇 시에요? B: 오후 2시입니다. A: 알겠습니다.", "내일 회의는 오후 2시입니다."),
            ("A: 점심 뭐 먹을까요? B: 한식이 좋겠어요. A: 좋아요.", "점심으로 한식을 먹기로 했습니다."),
            ("A: 오늘 날씨 어때? B: 맑고 좋아요.", "오늘 날씨는 맑고 좋습니다."),
        ],
        
        'role_generation': [
            ("[선생님] 파이썬이란 무엇인가요?", "파이썬은 배우기 쉬운 프로그래밍 언어입니다."),
            ("[친구] 주말에 뭐해?", "특별한 계획은 없어. 너는?"),
            ("[선생님] 인공지능의 장점은?", "인공지능은 대량의 데이터를 빠르게 처리할 수 있습니다."),
        ],
        
        'qa_generation': [
            ("질문: 서울의 인구는 얼마나 되나요?", "서울의 인구는 약 1천만 명입니다."),
            ("질문: 인공지능의 장점은 무엇인가요?", "인공지능은 대량의 데이터를 빠르게 처리하고 패턴을 학습할 수 있습니다."),
            ("질문: 한국의 수도는 어디인가요?", "한국의 수도는 서울입니다."),
        ]
    }
    
    # 각 태스크 테스트
    for task, test_cases in test_data.items():
        test_task(model, tokenizer, task, test_cases, device)
    
    # 전체 요약
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)
    
    print("\n💡 관찰 사항:")
    print("1. 학습 전과 비교해서 출력 품질 확인")
    print("2. 각 태스크별 성능 차이 분석")
    print("3. 더 많은 데이터와 에포크로 성능 향상 가능")
    print("4. 태스크별 fine-tuning으로 추가 개선 가능")
    
    print("\n📈 성능 향상 방법:")
    print("1. 더 많은 학습 데이터 사용 (현재: 태스크당 2개)")
    print("2. 더 많은 에포크 학습 (현재: 3 에포크)")
    print("3. Learning rate 조정")
    print("4. Beam search 파라미터 튜닝")
    print("5. 태스크별 가중치 조정")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()


