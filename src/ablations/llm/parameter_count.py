"""Count ChemLLM parameters from an actually loaded model, never its name."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .contracts import LLMAblationContractError, canonical_json_sha256


@dataclass(frozen=True, slots=True)
class ParameterCountReport:
    total_parameters: int
    trainable_parameters: int
    embedding_parameters: int
    non_embedding_parameters: int
    lora_trainable_parameters: int
    trainable_fraction: float
    dtype: tuple[str, ...]
    weight_bytes: int
    config_hidden_size: int | None
    num_layers: int | None
    num_attention_heads: int | None
    vocab_size: int | None
    source: str = "ACTUAL_LOADED_WEIGHTS"
    schema_version: str = "actual_parameter_count_report_v1"

    def __post_init__(self) -> None:
        if self.total_parameters <= 0:
            raise LLMAblationContractError("loaded model must contain parameters")
        if not 0 <= self.trainable_parameters <= self.total_parameters:
            raise LLMAblationContractError("invalid trainable parameter count")
        if self.embedding_parameters + self.non_embedding_parameters != self.total_parameters:
            raise LLMAblationContractError("embedding/non-embedding counts do not close")
        if not 0 <= self.lora_trainable_parameters <= self.trainable_parameters:
            raise LLMAblationContractError("invalid LoRA trainable count")
        if self.weight_bytes <= 0 or not self.dtype:
            raise LLMAblationContractError("weight bytes/dtype must come from loaded tensors")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dtype"] = list(self.dtype)
        payload["parameter_report_sha256"] = canonical_json_sha256(payload)
        return payload


def _config_value(config: object, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None and not isinstance(value, bool):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _parameter_identity(parameter: object) -> int:
    return id(parameter)


def _embedding_parameter_ids(model: object) -> set[int]:
    result: set[int] = set()
    for accessor in ("get_input_embeddings", "get_output_embeddings"):
        method = getattr(model, accessor, None)
        module = method() if callable(method) else None
        named = getattr(module, "named_parameters", None)
        if callable(named):
            for _, parameter in named(recurse=True):
                result.add(_parameter_identity(parameter))
    return result


def count_actual_loaded_parameters(model: object) -> ParameterCountReport:
    """Inspect the loaded tensor objects exposed by ``named_parameters``.

    Shared/tied tensors are counted once by object identity.  The function does
    not infer a count from a repository ID, configuration name, or marketing
    size label.
    """

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise LLMAblationContractError("model must expose named_parameters()")
    embedding_ids = _embedding_parameter_ids(model)
    seen: set[int] = set()
    total = trainable = embedding = lora_trainable = weight_bytes = 0
    dtypes: set[str] = set()
    for name, parameter in named_parameters():
        identity = _parameter_identity(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        numel = getattr(parameter, "numel", None)
        element_size = getattr(parameter, "element_size", None)
        if not callable(numel) or not callable(element_size):
            raise LLMAblationContractError(f"parameter {name} is not tensor-like")
        physical_count = int(numel())
        count = physical_count
        # bitsandbytes stores two logical 4-bit values per physical byte.
        # Its loaded quantization state retains the original tensor shape;
        # count that actual shape, not the packed storage element count.
        quant_state = getattr(parameter, "quant_state", None)
        logical_shape = getattr(quant_state, "shape", None)
        if logical_shape is not None:
            import math
            dimensions = tuple(int(value) for value in logical_shape)
            if not dimensions or any(value <= 0 for value in dimensions):
                raise LLMAblationContractError(f"parameter {name} has invalid loaded quantization shape")
            count = math.prod(dimensions)
        bytes_per_element = int(element_size())
        if count < 0 or bytes_per_element <= 0:
            raise LLMAblationContractError(f"parameter {name} has invalid shape/dtype")
        total += count
        weight_bytes += physical_count * bytes_per_element
        dtypes.add(str(getattr(parameter, "dtype", "unknown")))
        is_trainable = bool(getattr(parameter, "requires_grad", False))
        if is_trainable:
            trainable += count
            lowered = str(name).lower()
            if "lora_" in lowered or ".lora." in lowered:
                lora_trainable += count
        if identity in embedding_ids:
            embedding += count
    config = getattr(model, "config", object())
    return ParameterCountReport(
        total_parameters=total,
        trainable_parameters=trainable,
        embedding_parameters=embedding,
        non_embedding_parameters=total - embedding,
        lora_trainable_parameters=lora_trainable,
        trainable_fraction=(trainable / total if total else 0.0),
        dtype=tuple(sorted(dtypes)),
        weight_bytes=weight_bytes,
        config_hidden_size=_config_value(config, "hidden_size", "n_embd", "d_model"),
        num_layers=_config_value(config, "num_hidden_layers", "n_layer", "num_layers"),
        num_attention_heads=_config_value(
            config, "num_attention_heads", "n_head", "num_heads"
        ),
        vocab_size=_config_value(config, "vocab_size"),
    )


__all__ = ["ParameterCountReport", "count_actual_loaded_parameters"]
