from __future__ import annotations

from types import SimpleNamespace

from src.ablations.llm.parameter_count import count_actual_loaded_parameters


class FakeParameter:
    def __init__(self, count: int, *, trainable: bool, dtype: str = "torch.bfloat16"):
        self._count = count
        self.requires_grad = trainable
        self.dtype = dtype

    def numel(self) -> int:
        return self._count

    def element_size(self) -> int:
        return 2


class FakeEmbedding:
    def __init__(self, parameter):
        self.parameter = parameter

    def named_parameters(self, recurse=True):
        return [("weight", self.parameter)]


class FakeModel:
    def __init__(self):
        self.embedding = FakeParameter(100, trainable=False)
        self.base = FakeParameter(900, trainable=False)
        self.lora = FakeParameter(20, trainable=True)
        self.config = SimpleNamespace(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            vocab_size=100,
        )

    def get_input_embeddings(self):
        return FakeEmbedding(self.embedding)

    def get_output_embeddings(self):
        return FakeEmbedding(self.embedding)

    def named_parameters(self):
        return [
            ("model.embed_tokens.weight", self.embedding),
            ("model.layers.0.weight", self.base),
            ("model.layers.0.lora_A.weight", self.lora),
            ("lm_head.tied_weight", self.embedding),
        ]


def test_parameter_count_uses_loaded_unique_tensors() -> None:
    report = count_actual_loaded_parameters(FakeModel())
    assert report.total_parameters == 1020
    assert report.trainable_parameters == 20
    assert report.embedding_parameters == 100
    assert report.non_embedding_parameters == 920
    assert report.lora_trainable_parameters == 20
    assert report.weight_bytes == 2040
    assert report.config_hidden_size == 256
    assert report.source == "ACTUAL_LOADED_WEIGHTS"

