"""
Tiny student architecture tailored for knowledge distillation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartConfig, BartForConditionalGeneration


@dataclass
class TinyStudentConfig:
    """
    Helper dataclass that mirrors the important parts of BartConfig while
    keeping the tiny-student defaults in one place.
    """

    vocab_size: int = 8000
    max_position_embeddings: int = 256
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    encoder_attention_heads: int = 1
    decoder_attention_heads: int = 1
    encoder_ffn_dim: int = 256
    decoder_ffn_dim: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_function: str = "gelu"
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1
    layernorm_embedding: bool = True

    def to_bart_config(self) -> BartConfig:
        kwargs = asdict(self)
        kwargs["activation_dropout"] = kwargs.pop("attention_dropout")
        return BartConfig(
            vocab_size=kwargs["vocab_size"],
            max_position_embeddings=kwargs["max_position_embeddings"],
            d_model=kwargs["d_model"],
            encoder_layers=kwargs["encoder_layers"],
            decoder_layers=kwargs["decoder_layers"],
            encoder_attention_heads=kwargs["encoder_attention_heads"],
            decoder_attention_heads=kwargs["decoder_attention_heads"],
            encoder_ffn_dim=kwargs["encoder_ffn_dim"],
            decoder_ffn_dim=kwargs["decoder_ffn_dim"],
            dropout=kwargs["dropout"],
            attention_dropout=kwargs["attention_dropout"],
            activation_function=kwargs["activation_function"],
            pad_token_id=kwargs["pad_token_id"],
            bos_token_id=kwargs["bos_token_id"],
            eos_token_id=kwargs["eos_token_id"],
            scale_embedding=False,
            static_position_embeddings=False,
        )


class TiedLMHead(nn.Module):
    """
    Simple LM head that is explicitly tied to the shared embedding matrix.
    """

    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.embeddings.weight
        return F.linear(hidden_states, weight)


class TinyStudentForConditionalGeneration(nn.Module):
    """
    Minimal multi-head seq2seq model that mirrors the MultiTaskKoBART layout
    while using a tiny randomly initialised Bart backbone.
    """

    decoder_groups = {
        "shared_text": ["style_transfer", "dialogue_summarization", "role_generation"],
        "qa_generation": ["qa_generation"],
    }

    def __init__(
        self,
        config: Optional[TinyStudentConfig] = None,
        decoder_groups: Optional[Dict[str, list[str]]] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.student_config = config or TinyStudentConfig()
        bart_config = self.student_config.to_bart_config()

        self._decoder_groups = decoder_groups or self.decoder_groups
        self.task_to_decoder = {
            task: group for group, tasks in self._decoder_groups.items() for task in tasks
        }

        base_model = BartForConditionalGeneration(bart_config)
        if gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

        self.shared_encoder = base_model.model.encoder
        self.embeddings = base_model.model.shared

        self.decoders = nn.ModuleDict(
            {
                group_name: self._clone_decoder(base_model)
                for group_name in self._decoder_groups.keys()
            }
        )

        self.lm_heads = nn.ModuleDict(
            {
                group_name: TiedLMHead(self.embeddings)
                for group_name in self._decoder_groups.keys()
            }
        )

        self.dropout = nn.Dropout(bart_config.dropout)
        self.config = bart_config

    @staticmethod
    def _clone_decoder(base_model: BartForConditionalGeneration) -> nn.Module:
        decoder = copy.deepcopy(base_model.model.decoder)
        return decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        task: str = "style_transfer",
    ) -> Dict[str, torch.Tensor]:
        if task not in self.task_to_decoder:
            raise ValueError(f"Unknown task: {task}")

        decoder_key = self.task_to_decoder[task]
        encoder_outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        decoder = self.decoders[decoder_key]
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        hidden_states = self.dropout(decoder_outputs.last_hidden_state)
        logits = self.lm_heads[decoder_key](hidden_states)

        return {
            "logits": logits,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
            "decoder_hidden_states": decoder_outputs.last_hidden_state,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = "style_transfer",
        max_length: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = input_ids.device
        encoder_outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        decoder_key = self.task_to_decoder.get(task)
        if decoder_key is None:
            raise ValueError(f"Unknown task: {task}")

        decoder = self.decoders[decoder_key]
        lm_head = self.lm_heads[decoder_key]

        generated = torch.full(
            (input_ids.size(0), 1),
            fill_value=self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_length):
            decoder_outputs = decoder(
                input_ids=generated,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
            logits = lm_head(decoder_outputs.last_hidden_state)[:, -1, :] / temperature
            next_token = logits.softmax(dim=-1).argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break
        return generated

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())







