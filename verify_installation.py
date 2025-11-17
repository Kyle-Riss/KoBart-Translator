"""
KoBART 설치 검증 스크립트
"""

import sys

def check_imports():
    """필요한 라이브러리가 설치되었는지 확인"""
    print("라이브러리 확인 중...\n")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentencepiece': 'SentencePiece'
    }
    
    all_ok = True
    
    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:20} : {version}")
        except ImportError:
            print(f"✗ {name:20} : 설치되지 않음")
            all_ok = False
    
    return all_ok


def check_model():
    """모델 로드 가능 여부 확인"""
    print("\n" + "="*60)
    print("모델 로드 테스트...\n")
    
    try:
        from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
        
        print("토크나이저 로딩...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        print("✓ 토크나이저 로드 성공")
        
        print("\n모델 로딩...")
        model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
        print("✓ 모델 로드 성공")
        
        # 간단한 테스트
        print("\n간단한 생성 테스트...")
        test_input = "테스트입니다."
        inputs = tokenizer(test_input, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=20)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"입력: {test_input}")
        print(f"출력: {result}")
        print("✓ 모델 작동 확인")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        return False


def check_device():
    """사용 가능한 디바이스 확인"""
    print("\n" + "="*60)
    print("디바이스 정보...\n")
    
    try:
        import torch
        
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CPU 모드로 실행됩니다.")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Apple Silicon (MPS) 사용 가능")
        
    except Exception as e:
        print(f"디바이스 정보를 가져올 수 없습니다: {e}")


def main():
    """메인 검증 함수"""
    print("="*60)
    print("KoBART 설치 검증")
    print("="*60)
    print()
    
    # Python 버전 확인
    print(f"Python 버전: {sys.version}")
    print()
    
    # 라이브러리 확인
    step1 = check_imports()
    
    if not step1:
        print("\n⚠️ 필요한 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("  pip3 install -r requirements.txt")
        return
    
    # 디바이스 확인
    check_device()
    
    # 모델 로드 확인
    step2 = check_model()
    
    # 최종 결과
    print("\n" + "="*60)
    if step1 and step2:
        print("✅ 모든 검증 완료! KoBART를 사용할 준비가 되었습니다.")
        print("\n다음 스크립트로 시작하세요:")
        print("  python3 quick_start.py")
        print("  python3 example_simple.py")
    else:
        print("⚠️ 일부 검증에 실패했습니다.")
        print("USAGE_GUIDE.md를 참고하세요.")
    print("="*60)


if __name__ == "__main__":
    main()


