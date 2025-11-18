"""
학습된 Multi-Task KoBART 대화형 데모
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from kobart_translator import MultiTaskKoBART
from transformers import PreTrainedTokenizerFast


class InteractiveDemo:
    """대화형 데모 클래스"""
    
    def __init__(self, checkpoint_path: str):
        """초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*60)
        print("Multi-Task KoBART 대화형 데모")
        print("="*60)
        print()
        
        # 토크나이저 로드
        print("토크나이저 로딩 중...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        print("✓ 토크나이저 로드 완료")
        
        # 모델 로드
        print("\n학습된 모델 로딩 중...")
        self.model = MultiTaskKoBART()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("✓ 학습된 모델 로드 완료")
        
        # 태스크 정보
        self.tasks = {
            '1': ('style_transfer', '스타일 변환 (구어체 → 격식체)'),
            '2': ('dialogue_summarization', '대화 요약'),
            '3': ('role_generation', '역할 기반 응답 생성'),
            '4': ('qa_generation', 'QA 답변 생성')
        }
        
        print(f"\n디바이스: {self.device}")
        print("준비 완료!\n")
    
    def show_menu(self):
        """메뉴 표시"""
        print("\n" + "="*60)
        print("태스크 선택:")
        print("-"*60)
        for key, (task_id, description) in self.tasks.items():
            print(f"  {key}. {description}")
        print("  0. 종료")
        print("="*60)
    
    def generate(self, text: str, task: str) -> str:
        """텍스트 생성"""
        with torch.no_grad():
            # 토큰화
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task=task,
                max_length=100
            )
            
            # 디코딩
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
    
    def run(self):
        """대화형 모드 실행"""
        print("대화형 모드 시작!")
        print("(각 태스크를 선택하고 텍스트를 입력하세요)\n")
        
        while True:
            self.show_menu()
            
            choice = input("\n태스크 번호 입력: ").strip()
            
            if choice == '0':
                print("\n👋 프로그램을 종료합니다. 감사합니다!")
                break
            
            if choice not in self.tasks:
                print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
                continue
            
            task_id, task_name = self.tasks[choice]
            print(f"\n선택된 태스크: {task_name}")
            print("-"*60)
            
            # 태스크별 입력 예시
            examples = {
                'style_transfer': '예: "이거 좀 도와주세요"',
                'dialogue_summarization': '예: "A: 회의 몇시? B: 2시요"',
                'role_generation': '예: "[선생님] 파이썬이란?"',
                'qa_generation': '예: "질문: 서울 인구는?"'
            }
            
            print(f"입력 형식: {examples[task_id]}")
            user_input = input("\n입력: ").strip()
            
            if not user_input:
                print("⚠️ 입력이 비어있습니다.")
                continue
            
            print("\n생성 중...", end='', flush=True)
            
            try:
                result = self.generate(user_input, task_id)
                print("\r" + " "*20)  # 지우기
                print(f"출력: {result}")
                
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
            
            print("\n" + "-"*60)
            input("Enter 키를 눌러 계속...")


def main():
    """메인 함수"""
    checkpoint_path = "/Users/arka/Desktop/Ko-bart/multi_task_model.pt"
    
    try:
        demo = InteractiveDemo(checkpoint_path)
        demo.run()
        
    except FileNotFoundError:
        print("❌ 학습된 모델을 찾을 수 없습니다.")
        print("먼저 train_multi_task.py를 실행하여 모델을 학습시켜주세요.")
        print("\n실행 방법:")
        print("  python3 train_multi_task.py")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()


