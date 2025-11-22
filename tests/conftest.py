import torch
import pytest


class SimpleTokenizer:
    """
    최소한의 토크나이저 구현.
    - 공백 단위로 토큰화
    - 새로운 토큰은 자동으로 vocab에 추가
    - HuggingFace 토크나이저의 핵심 API만 흉내냄
    """

    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.token_to_id = {
            self.pad_token: self.pad_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
        }
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    def _encode_tokens(self, text: str):
        tokens = text.strip().split()
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token
            ids.append(self.token_to_id[token])
        return ids

    def _pad(self, ids, max_length):
        attention = [1] * len(ids)
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
                attention = attention[:max_length]
            elif len(ids) < max_length:
                pad_length = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_length
                attention = attention + [0] * pad_length
        return ids, attention

    def _build_inputs(self, text, max_length, truncation):
        if isinstance(text, list):
            sequences = [
                self._build_inputs(single_text, max_length, truncation)
                for single_text in text
            ]
            input_ids = torch.stack([seq["input_ids"][0] for seq in sequences], dim=0)
            attention_mask = torch.stack(
                [seq["attention_mask"][0] for seq in sequences], dim=0
            )
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if text is None:
            text = ""

        ids = [self.bos_token_id] + self._encode_tokens(text) + [self.eos_token_id]
        if max_length is None and truncation:
            max_length = 512
        ids, attention = self._pad(ids, max_length)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }

    def __call__(
        self,
        text=None,
        *,
        text_target=None,
        return_tensors="pt",
        max_length=None,
        truncation=False,
        padding=False,
    ):
        target_text = text if text_target is None else text_target
        outputs = self._build_inputs(
            target_text, max_length=max_length, truncation=truncation
        )

        if padding and max_length is None:
            # padding=True 이고 max_length가 없으면 가장 긴 시퀀스에 맞춰 패딩
            sequences = outputs["input_ids"]
            max_len = sequences.size(1)
            outputs = self._build_inputs(target_text, max_length=max_len, truncation=False)

        if return_tensors == "pt":
            return outputs
        raise ValueError("Only return_tensors='pt' is supported in SimpleTokenizer.")

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for idx in token_ids:
            idx = idx.item() if hasattr(idx, "item") else idx
            token = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in {
                self.pad_token,
                self.bos_token,
                self.eos_token,
            }:
                continue
            tokens.append(token)
        return " ".join(tokens).strip()


class DummyMultiTaskModel(torch.nn.Module):
    """
    MultiTaskKoBART 대체용 경량 모형.
    - 입력을 그대로 반환하여 generate 인터페이스만 만족.
    """

    def __init__(self, tokenizer: SimpleTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Dummy model은 forward를 사용하지 않습니다.")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task: str = "style_transfer",
        max_length: int = 64,
    ) -> torch.Tensor:
        del attention_mask, task, max_length

        batch_outputs = []
        for sequence in input_ids:
            seq = sequence
            if not (seq == self.tokenizer.eos_token_id).any():
                seq = torch.cat(
                    [
                        seq,
                        torch.tensor(
                            [self.tokenizer.eos_token_id],
                            dtype=seq.dtype,
                            device=seq.device,
                        ),
                    ],
                    dim=0,
                )
            batch_outputs.append(seq)

        max_len = max(seq.size(0) for seq in batch_outputs)
        padded = []
        for seq in batch_outputs:
            if seq.size(0) < max_len:
                pad_len = max_len - seq.size(0)
                pad = torch.full(
                    (pad_len,),
                    self.tokenizer.pad_token_id,
                    dtype=seq.dtype,
                    device=seq.device,
                )
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)

        return torch.stack(padded, dim=0)


@pytest.fixture(scope="session")
def tokenizer():
    return SimpleTokenizer()


@pytest.fixture(scope="session")
def model(tokenizer):
    return DummyMultiTaskModel(tokenizer)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


TASK_TEST_CASES = {
    "style_transfer": [
        ("[ban→yo] 안녕?", "안녕하세요?"),
        ("[yo→sho] 확인해주세요.", "확인해주십시오."),
    ],
    "dialogue_summarization": [
        ("A: 안녕\nB: 안녕하세요", "둘이 인사함"),
        ("A: 밥 먹었어?\nB: 아직이야", "아직 식사 안 함"),
    ],
    "role_generation": [
        ("[의사] 어디가 아프세요?", "증상을 말해 주세요."),
        ("[친구] 주말에 뭐해?", "아직 계획 없어."),
    ],
    "qa_generation": [
        ("[DOC] 서울은 한국의 수도이다. [Q] 한국의 수도는?", "서울"),
        ("[DOC] 사과는 과일이다. [Q] 사과는 무엇인가?", "과일"),
    ],
}


@pytest.fixture(scope="session", params=list(TASK_TEST_CASES.keys()))
def task(request):
    return request.param


@pytest.fixture(scope="session")
def test_cases(task):
    return TASK_TEST_CASES[task]




