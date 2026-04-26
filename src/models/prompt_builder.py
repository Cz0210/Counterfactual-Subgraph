"""Prompt templates for ChemLLM counterfactual subgraph generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """One hard-coded prompt example for ChemLLM."""

    parent_smiles: str
    fragment_smiles: str
    rationale: str = ""


SYSTEM_PROMPT = (
    "You are ChemLLM for counterfactual subgraph generation.\n"
    "Given one complete parent molecule SMILES, output exactly one connected "
    "subgraph SMILES from that parent.\n"
    "Output the core fragment only and do not use dummy atoms such as '*'.\n"
    "The fragment must stay chemically meaningful, connected, and must be a "
    "real subgraph of the parent molecule.\n"
    "Output the fragment SMILES only. Do not explain, do not restate the "
    "parent, and do not add any extra words."
)


FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        parent_smiles="Cc1ccccc1",
        fragment_smiles="C",
        rationale="methyl branch from toluene",
    ),
    FewShotExample(
        parent_smiles="COC",
        fragment_smiles="CO",
        rationale="methoxy fragment from dimethyl ether",
    ),
    FewShotExample(
        parent_smiles="c1ccccc1O",
        fragment_smiles="O",
        rationale="hydroxyl group from phenol",
    ),
)


def build_counterfactual_system_prompt() -> str:
    """Return the system prompt used for ChemLLM inference."""

    return SYSTEM_PROMPT


def format_few_shot_examples() -> str:
    """Render the hard-coded few-shot examples for the prompt body."""

    blocks: list[str] = []
    for index, example in enumerate(FEW_SHOT_EXAMPLES, start=1):
        blocks.extend(
            [
                f"Example {index}",
                f"PARENT_SMILES: {example.parent_smiles}",
                f"FRAGMENT_SMILES: {example.fragment_smiles}",
            ]
        )
        if example.rationale:
            blocks.append(f"NOTE: {example.rationale}")
        blocks.append("")
    return "\n".join(blocks).strip()


def build_chemllm_prompt(
    parent_smiles: str,
    *,
    label: int | None = None,
) -> str:
    """Build the plain-text few-shot prompt for ChemLLM."""

    lines = [
        "[System]",
        build_counterfactual_system_prompt(),
        "",
        "[Examples]",
        format_few_shot_examples(),
        "",
        "[Task]",
    ]
    if label is not None:
        lines.append(f"ORIGINAL_LABEL: {label}")
    lines.extend(
        [
            f"PARENT_SMILES: {str(parent_smiles).strip()}",
            "FRAGMENT_SMILES:",
        ]
    )
    return "\n".join(lines)


def build_chemllm_messages(
    parent_smiles: str,
    *,
    label: int | None = None,
) -> list[dict[str, str]]:
    """Build a message list that can be passed to chat-style tokenizers."""

    user_lines = ["Return exactly one connected core fragment SMILES without dummy atoms."]
    if label is not None:
        user_lines.append(f"ORIGINAL_LABEL: {label}")
    user_lines.extend(
        [
            f"PARENT_SMILES: {str(parent_smiles).strip()}",
            "FRAGMENT_SMILES:",
        ]
    )
    return [
        {"role": "system", "content": build_counterfactual_system_prompt()},
        {
            "role": "user",
            "content": "[Examples]\n"
            + format_few_shot_examples()
            + "\n\n[Task]\n"
            + "\n".join(user_lines),
        },
    ]
