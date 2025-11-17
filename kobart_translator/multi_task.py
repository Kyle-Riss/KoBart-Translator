"""
Multi-Task KoBART Architecture
- Shared Encoder (1개)
- 4개의 Task-specific Decoder Heads
"""

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from typing import Dict, Optional


class MultiTaskKoBART(nn.Module):
    """
    KoBART 기반 멀티태스크 모델
    
    구조:
    - Shared Encoder (KoBART 인코더)
    - 4개의 Decoder Heads:
        1. Style Transfer
        2. Dialogue Summarization
        3. Role-conditioned Generation
        4. QA Answer Generation
    """
    
    def __init__(self, model_name: str = 'gogamza/kobart-base-v1'):
        super(MultiTaskKoBART, self).__init__()
        
        print("Multi-Task KoBART 모델 초기화 중...")
        
        # 기본 KoBART 모델 로드
        base_model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Shared Encoder (공유 인코더)
        self.shared_encoder = base_model.model.encoder
        print("[OK] Shared Encoder 로드 완료")
        
        # 기본 디코더 설정 정보 가져오기
        decoder_config = base_model.model.decoder.config
        
        # 4개의 Task-specific Decoder Heads
        # 각 디코더는 동일한 구조를 가지지만 독립적으로 학습됨
        self.decoders = nn.ModuleDict({
            'style_transfer': self._create_decoder(base_model),
            'dialogue_summarization': self._create_decoder(base_model),
            'role_generation': self._create_decoder(base_model),
            'qa_generation': self._create_decoder(base_model)
        })
        
        # Language Model Heads (각 디코더용)
        vocab_size = base_model.config.vocab_size
        hidden_size = base_model.config.d_model
        
        self.lm_heads = nn.ModuleDict({
            'style_transfer': nn.Linear(hidden_size, vocab_size, bias=False),
            'dialogue_summarization': nn.Linear(hidden_size, vocab_size, bias=False),
            'role_generation': nn.Linear(hidden_size, vocab_size, bias=False),
            'qa_generation': nn.Linear(hidden_size, vocab_size, bias=False)
        })
        
        print("[OK] 4개의 Decoder Heads 생성 완료:")
        print("  - Head 1: Style Transfer")
        print("  - Head 2: Dialogue Summarization")
        print("  - Head 3: Role-conditioned Generation")
        print("  - Head 4: QA Answer Generation")
        
        self.config = base_model.config
        
    def _create_decoder(self, base_model):
        """기본 모델의 디코더를 복사하여 새로운 디코더 생성"""
        import copy
        return copy.deepcopy(base_model.model.decoder)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        task: str = 'style_transfer'
    ):
        """
        Forward pass
        
        Args:
            input_ids: 입력 토큰 IDs
            attention_mask: 어텐션 마스크
            decoder_input_ids: 디코더 입력 IDs
            task: 태스크 이름 ('style_transfer', 'dialogue_summarization', 
                              'role_generation', 'qa_generation')
        """
        # Shared Encoder로 인코딩
        encoder_outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Task-specific Decoder로 디코딩
        if task not in self.decoders:
            raise ValueError(f"Unknown task: {task}")
        
        decoder = self.decoders[task]
        lm_head = self.lm_heads[task]
        
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask
        )
        
        # LM Head를 통해 최종 logits 생성
        logits = lm_head(decoder_outputs.last_hidden_state)
        
        return {
            'logits': logits,
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = 'style_transfer',
        max_length: int = 50,
        num_beams: int = 5, # 구현 필요
        **kwargs
    ):
        """
        텍스트 생성
        
        Args:
            input_ids: 입력 토큰 IDs
            attention_mask: 어텐션 마스크
            task: 태스크 이름
            max_length: 최대 생성 길이
            num_beams: Beam search 크기
        """
        # Shared Encoder로 인코딩
        encoder_outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Task-specific Decoder 선택
        decoder = self.decoders[task]
        lm_head = self.lm_heads[task]
        
        # 간단한 greedy decoding (실제로는 beam search 구현 필요)
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 시작 토큰 (BOS)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device
        )
        
        generated = decoder_input_ids
        
        for _ in range(max_length):
            decoder_outputs = decoder(
                input_ids=generated,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask
            )
            
            logits = lm_head(decoder_outputs.last_hidden_state)
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # EOS 토큰이면 종료
            if (next_tokens == self.config.eos_token_id).all():
                break
        
        return generated
    
    def get_encoder_parameters(self):
        """공유 인코더의 파라미터 반환"""
        return self.shared_encoder.parameters()
    
    def get_decoder_parameters(self, task: str):
        """특정 태스크 디코더의 파라미터 반환"""
        if task not in self.decoders:
            raise ValueError(f"Unknown task: {task}")
        return list(self.decoders[task].parameters()) + list(self.lm_heads[task].parameters())
    
    def freeze_encoder(self):
        """인코더 파라미터 고정"""
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        print("[OK] Encoder 파라미터 고정")
    
    def unfreeze_encoder(self):
        """인코더 파라미터 해제"""
        for param in self.shared_encoder.parameters():
            param.requires_grad = True
        print("[OK] Encoder 파라미터 학습 가능")


def main():
    """테스트 및 사용 예제"""
    print("="*60)
    print("Multi-Task KoBART 모델 테스트")
    print("="*60)
    print()
    
    # 모델 및 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    model = MultiTaskKoBART()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"\n[OK] 모델 로드 완료 (디바이스: {device})")
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.shared_encoder.parameters())
    
    print(f"\n모델 정보:")
    print(f"  - 전체 파라미터: {total_params:,}")
    print(f"  - 공유 인코더 파라미터: {encoder_params:,}")
    print(f"  - 디코더 헤드 개수: 4개")
    
    # 각 태스크별 테스트
    test_cases = {
        'style_transfer': "이 문장을 격식있는 표현으로 바꿔주세요.",
        'dialogue_summarization': "A: 안녕하세요. B: 네, 안녕하세요. 무엇을 도와드릴까요?",
        'role_generation': "선생님 역할로 설명해주세요: 인공지능이란 무엇인가요?",
        'qa_generation': "질문: 한국의 수도는 어디인가요?"
    }
    
    print("\n" + "="*60)
    print("태스크별 생성 테스트")
    print("="*60)
    
    with torch.no_grad():
        for task, text in test_cases.items():
            print(f"\n[{task.upper()}]")
            print(f"입력: {text}")
            
            # 토큰화
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 생성
            try:
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    task=task,
                    max_length=50
                )
                
                # 디코딩
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"출력: {result}")
            except Exception as e:
                print(f"생성 중 오류: {e}")
    
    print("\n" + "="*60)
    print("✓ 테스트 완료!")
    print("="*60)
    
    # 사용 팁
    print("\n💡 사용 팁:")
    print("1. 각 태스크별로 독립적으로 fine-tuning 가능")
    print("2. 인코더는 모든 태스크에서 공유되어 효율적")
    print("3. model.freeze_encoder()로 인코더 고정 가능")
    print("4. 태스크별 학습 데이터로 각 디코더 헤드 학습")


if __name__ == "__main__":
    main()

