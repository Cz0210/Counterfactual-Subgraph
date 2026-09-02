"""Render claim-safe paper templates for ablations with no result values yet."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Mapping


TEMPLATES: Mapping[str, str] = {
    "llm_ablation_section_template.tex": r"""% AUTO-GENERATED DESIGN TEMPLATE; NO RESULTS HAVE BEEN INSERTED.
\subsection{Proposal-generator ablation}
\label{sec:llm-proposer-ablation}
The registered design contains a train-only fixed BRICS control and ChemLLM
off-the-shelf, matched-SFT, and matched-SFT-plus-PPO slots.  A slot is evaluated
only when its exact BACE checkpoint lineage is available.  At framework-build
time no independently matched BACE SFT checkpoint was verified, so the SFT and
SFT-plus-PPO slots remain blocked rather than being substituted.  Any future
available variants must use the same attempt-matched budget and frozen
BACE/Ours parser, verifier, selector, oracle, and test evaluator.

The intended, not-yet-tested contrasts are BRICS versus off-the-shelf proposal
prior, off-the-shelf versus supervised task adaptation, and SFT versus PPO
counterfactual alignment.  These labels describe the experimental design and
are not findings.

% TODO(result): insert hash-bound aggregate rows after the authorized runs pass.
% TODO(claim): phrase conclusions only to the strength supported by bootstrap CIs.
""",
    "llm_ablation_table_template.tex": r"""% AUTO-GENERATED TEMPLATE; DO NOT REPLACE TODOs WITHOUT A PASS MANIFEST.
\begin{table}[t]
  \centering
  \caption{BACE proposal-generator ablation under a shared downstream protocol.}
  \label{tab:llm-proposer-ablation}
  \begin{tabular}{llrrrr}
    \toprule
    Variant & Availability & Valid rate & Strict flip rate & CCRCov@10 & Median WNode \\
    \midrule
    BRICS fixed & CONFIG ONLY & TODO & TODO & TODO & TODO \\
    ChemLLM off-the-shelf & CONFIG ONLY & TODO & TODO & TODO & TODO \\
    ChemLLM SFT & BLOCKED: no matched checkpoint & -- & -- & -- & -- \\
    ChemLLM SFT+PPO & BLOCKED: no matched SFT lineage & -- & -- & -- & -- \\
    \bottomrule
  \end{tabular}
\end{table}
""",
    "gnn_ablation_section_template.tex": r"""% AUTO-GENERATED DESIGN TEMPLATE; NO RESULTS HAVE BEEN INSERTED.
\subsection{Classifier-backbone ablation}
\label{sec:gnn-backbone-ablation}
We evaluate GINE, GIN, GCN, and GATv2 on BACE using identical split roles and
feature-schema checks.  The primary proposal-fixed protocol reuses canonical
candidate identities from BACE/Ours and recomputes classifier-dependent source
cohorts, strict flips, WNode matrices, calibration selection, and held-out test
evaluation.  We report both native cohorts and the common correctly classified
parent intersection.  End-to-end reruns are an optional, separately authorized
configuration.

% TODO(result): insert only values linked to a PASS aggregate manifest.
""",
    "gnn_ablation_table_template.tex": r"""% AUTO-GENERATED TEMPLATE; DO NOT REPLACE TODOs WITHOUT A PASS MANIFEST.
\begin{table}[t]
  \centering
  \caption{BACE backbone ablation on native and common source cohorts.}
  \label{tab:gnn-backbone-ablation}
  \begin{tabular}{lrrrr}
    \toprule
    Backbone & ROC-AUC & ECE & Common CCRCov@10 & Common median WNode \\
    \midrule
    GINE & TODO & TODO & TODO & TODO \\
    GIN & TODO & TODO & TODO & TODO \\
    GCN & TODO & TODO & TODO & TODO \\
    GATv2 & TODO & TODO & TODO & TODO \\
    \bottomrule
  \end{tabular}
\end{table}
""",
    "ablation_claims_checklist.md": """# Ablation claims checklist

- [ ] Every numeric value resolves to a PASS run and aggregate manifest hash.
- [ ] BRICS vocabulary provenance is BACE train only; calibration/test are absent.
- [ ] All proposal variants use the same attempt-matched primary budget.
- [ ] Base, SFT, and PPO checkpoint identities match the frozen BACE reference.
- [ ] A missing matched SFT checkpoint remains blocked; no substitute is used.
- [ ] Verifier, selector, thresholds, GINE oracle, and evaluator are shared.
- [ ] GNN checkpoint selection uses validation only and test is final-report only.
- [ ] Native and common source cohorts are both reported.
- [ ] GINE/GIN/GCN/GATv2 edge-feature modes are disclosed explicitly.
- [ ] Confidence intervals are parent-bootstrap estimates with seeds recorded.
- [ ] No causal or significance language appears without supporting analysis.
- [ ] The final prose does not claim that external knowledge was proven.
""",
    "llm_stage_ablation_section.tex": r"""% DESIGN TEMPLATE ONLY; NO RESULT VALUES.
\subsection{ChemLLM adaptation-stage ablation}
\label{sec:llm-stage-ablation}
We register an attempt-matched train-only BRICS control, an off-the-shelf
ChemLLM proposal model, a project-SFT slot, and a project-SFT-plus-PPO slot.
Every available row uses the same BACE parent cohort, proposal budget, parser,
projection, frozen GINE verifier, calibration selector, and held-out test
evaluator.  The frozen BACE main provenance currently records a fresh LoRA
initializer followed by PPO, not an independently matched project-SFT stage;
therefore the SFT and SFT-plus-PPO labels remain blocked until exact matching
checkpoints are demonstrated.

% TODO(result): insert only hash-bound PASS aggregates.
""",
    "llm_scale_sensitivity_section.tex": r"""% DESIGN TEMPLATE ONLY; NO RESULT VALUES.
\subsection{ChemLLM model-scale sensitivity}
\label{sec:llm-scale-sensitivity}
The primary registered contrast changes only the base-model scale between a
2B and 7B project-SFT-plus-PPO pipeline.  It may run only if both adaptation
chains match in SFT data and optimizer updates, PPO parents/reward/updates,
proposal budget, and downstream evaluation.  If that evidence is unavailable
but prompt and tokenizer contracts are comparable, the claim is narrowed to
off-the-shelf proposal sensitivity.  The registered 20B model is metadata-only
and is not part of the current experiment.

% TODO(result): insert only hash-bound PASS aggregates.
""",
    "llm_ablation_claim_checklist.md": """# LLM stage/scale claim checklist

- [ ] Stage and scale effects are reported in separate tables.
- [ ] A1 is called off-the-shelf, not raw pretrained.
- [ ] A2 binds an exact BACE project-SFT checkpoint and never loads PPO.
- [ ] A3 binds exact project-SFT and PPO checkpoints.
- [ ] The observed BACE fresh-LoRA-plus-PPO lineage is not relabeled SFT+PPO.
- [ ] The 2B/7B full-method comparison matches SFT/PPO data and update counts.
- [ ] Proposal-only fallback is explicitly labeled model-scale sensitivity.
- [ ] Parameter counts come from loaded tensors, not 2B/7B/20B names.
- [ ] BRICS vocabulary and proposals use BACE train only.
- [ ] Calibration/test are absent from SFT and PPO.
- [ ] The 20B weights were neither downloaded nor executed.
- [ ] Bootstrap uses test parents only after selector freeze and never selects a variant.
""",
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_paper_staging(output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    written: list[Path] = []
    for filename, content in TEMPLATES.items():
        path = root / filename
        _atomic_text(path, content)
        written.append(path)
    return written


__all__ = ["TEMPLATES", "build_paper_staging"]
