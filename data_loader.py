"""
어체 변환 데이터 로더
"""

import os
from typing import List, Dict, Tuple
import random


class StyleTransferDataLoader:
    """어체 변환 데이터 로더"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.parallel_dir = os.path.join(base_dir, "수동태깅 병렬데이터")
        self.opus_dir = os.path.join(base_dir, "OPUS 오픈 코퍼스 단일데이터")
        
    def load_parallel_data(
        self,
        split: str = 'train',
        style_pairs: List[Tuple[str, str]] = None,
        max_samples: int = None
    ) -> List[Dict]:
        """
        병렬 데이터 로드
        
        Args:
            split: 'train', 'dev', 'test'
            style_pairs: [('ban', 'yo'), ('ban', 'sho'), ('yo', 'sho')]
            max_samples: 최대 샘플 수 (None이면 전체)
        
        Returns:
            [{'input': '응', 'target': '네', 'source_style': 'ban', 'target_style': 'yo'}, ...]
        """
        if style_pairs is None:
            # 모든 가능한 조합
            style_pairs = [
                ('ban', 'yo'),
                ('ban', 'sho'),
                ('yo', 'ban'),
                ('yo', 'sho'),
                ('sho', 'ban'),
                ('sho', 'yo')
            ]
        
        # 파일명 결정
        if split == 'train':
            prefix = 'train_ext'
        else:
            prefix = split
        
        # 데이터 로드
        data = []
        
        # 각 스타일 파일 로드
        styles = {}
        for style in ['ban', 'yo', 'sho']:
            file_path = os.path.join(self.parallel_dir, f"{prefix}.{style}.txt")
            with open(file_path, 'r', encoding='utf-8') as f:
                styles[style] = [line.strip() for line in f if line.strip()]
        
        # 병렬 데이터 생성
        num_lines = len(styles['ban'])
        
        # max_samples 적용 (line 수를 제한)
        if max_samples is not None:
            num_lines = min(num_lines, max_samples // len(style_pairs))
        
        for i in range(num_lines):
            for source_style, target_style in style_pairs:
                if styles[source_style][i] and styles[target_style][i]:
                    data.append({
                        'task': 'style_transfer',
                        'input': styles[source_style][i],
                        'target': styles[target_style][i],
                        'source_style': source_style,
                        'target_style': target_style
                    })
        
        return data
    
    def load_opus_data(
        self,
        max_samples: int = 10000,
        styles: List[str] = None
    ) -> List[Dict]:
        """
        OPUS 오픈 코퍼스 데이터 로드 (단일 스타일)
        
        Args:
            max_samples: 최대 샘플 수
            styles: ['ban', 'yo', 'sho'] 중 선택
        
        Returns:
            데이터 리스트
        """
        if styles is None:
            styles = ['ban', 'yo', 'sho']
        
        data = []
        
        for style in styles:
            file_path = os.path.join(self.opus_dir, f"opensubtitles.{style}.txt")
            
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # 샘플링
            if max_samples is not None and len(lines) > max_samples:
                lines = random.sample(lines, max_samples)
            
            # 역변환 태스크로 생성 (자기 자신으로 변환)
            for line in lines:
                data.append({
                    'task': 'style_transfer',
                    'input': line,
                    'target': line,  # 같은 스타일 유지
                    'source_style': style,
                    'target_style': style
                })
        
        return data
    
    def get_train_data(
        self,
        use_parallel: bool = True,
        use_opus: bool = False,
        opus_samples: int = 10000,
        parallel_samples: int = None
    ) -> List[Dict]:
        """학습 데이터 생성"""
        data = []
        
        if use_parallel:
            print("병렬 데이터 로딩 중...")
            parallel_data = self.load_parallel_data('train', max_samples=parallel_samples)
            data.extend(parallel_data)
            print(f"[OK] 병렬 데이터: {len(parallel_data):,}개")
        
        if use_opus:
            print("OPUS 데이터 로딩 중...")
            opus_data = self.load_opus_data(max_samples=opus_samples)
            data.extend(opus_data)
            print(f"[OK] OPUS 데이터: {len(opus_data):,}개")
        
        # 셔플
        random.shuffle(data)
        
        return data
    
    def get_dev_data(self) -> List[Dict]:
        """검증 데이터 생성"""
        return self.load_parallel_data('dev')
    
    def get_test_data(self) -> List[Dict]:
        """테스트 데이터 생성"""
        return self.load_parallel_data('test')
    
    def print_statistics(self):
        """데이터 통계 출력"""
        print("="*60)
        print("데이터 통계")
        print("="*60)
        
        # 병렬 데이터
        train_data = self.load_parallel_data('train')
        dev_data = self.load_parallel_data('dev')
        test_data = self.load_parallel_data('test')
        
        print(f"\n병렬 데이터:")
        print(f"  Train: {len(train_data):,}개")
        print(f"  Dev:   {len(dev_data):,}개")
        print(f"  Test:  {len(test_data):,}개")
        
        # 스타일별 분포
        style_counts = {}
        for item in train_data:
            pair = f"{item['source_style']} → {item['target_style']}"
            style_counts[pair] = style_counts.get(pair, 0) + 1
        
        print(f"\n스타일 변환 쌍:")
        for pair, count in sorted(style_counts.items()):
            print(f"  {pair}: {count:,}개")
        
        # OPUS 데이터 (파일 크기만)
        print(f"\nOPUS 코퍼스 (파일 수):")
        for style in ['ban', 'yo', 'sho']:
            file_path = os.path.join(self.opus_dir, f"opensubtitles.{style}.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    num_lines = sum(1 for line in f if line.strip())
                print(f"  {style}: {num_lines:,}개")
        
        print("="*60)


def main():
    """테스트 함수"""
    loader = StyleTransferDataLoader()
    
    # 통계 출력
    loader.print_statistics()
    
    print("\n샘플 데이터:")
    print("-"*60)
    
    # 학습 데이터 샘플
    train_data = loader.get_train_data(use_parallel=True, use_opus=False)
    
    print(f"\n총 학습 데이터: {len(train_data):,}개")
    print("\n첫 5개 샘플:")
    for i, item in enumerate(train_data[:5], 1):
        print(f"\n{i}. {item['source_style']} → {item['target_style']}")
        print(f"   입력: {item['input']}")
        print(f"   출력: {item['target']}")


if __name__ == "__main__":
    main()


