"""基于 RDKit 的化学规则奖励计算。

这个模块只负责两类结构性奖励：

1. `R_valid`：生成结果能否被 RDKit 正常解析与 sanitize；
2. `R_subgraph`：生成片段是否真的是原分子的一个子图。

注意：
- 这里显式支持带 `*` 的 capped fragment；
- `*` 在奖励逻辑中表示切断边界的 dummy atom，而不是“要被删除的真实原子”；
- 对于带 `*` 的片段，我们会先构建 query，再做更严格的边界覆盖检查，
  以避免 `HasSubstructMatch` 因 wildcard 语义带来假阳性。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chem.smiles_utils import is_rdkit_available, parse_smiles
from src.chem.substructure import is_connected_fragment, is_valid_capped_subgraph

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - 取决于本地环境
    Chem = None


@dataclass(slots=True)
class ChemRewardEngine:
    """化学规则打分器。

    返回值约定与用户需求保持一致：
    - 通过返回 `1.0`
    - 不通过返回 `-1.0`
    """

    valid_pass_reward: float = 1.0
    valid_fail_reward: float = -1.0
    subgraph_pass_reward: float = 1.0
    subgraph_fail_reward: float = -1.0

    def check_validity(self, generated_smiles: str) -> float:
        """检查生成片段是否是可解析、可 sanitize 的 SMILES。

        这里允许带 dummy atom 的 capped fragment，因为在本项目里 `*`
        是切断边界的合法表示形式。
        """

        parsed = parse_smiles(
            generated_smiles,
            sanitize=True,
            canonicalize=False,
            allow_capped_fragments=True,
        )
        if parsed.sanitized:
            return self.valid_pass_reward
        return self.valid_fail_reward

    def check_subgraph(self, original_smiles: str, generated_smiles: str) -> float:
        """检查生成片段是否真实存在于原分子中。

        对不带 `*` 的普通片段：
        - 直接使用 `HasSubstructMatch(fragment_mol)`。

        对带 `*` 的 capped fragment：
        1. 使用 `Chem.MolFromSmarts(...)` 构建 query；
        2. 先用 `HasSubstructMatch(query)` 做快速存在性判断；
        3. 再调用更严格的 `is_valid_capped_subgraph(...)` 做边界覆盖校验。

        第 3 步非常重要，因为单纯的 wildcard query 可能把内部原子也匹配上，
        从而把“看起来像子图、但并不是合法 cut boundary”的幻觉结果误判为真。
        """

        if not is_rdkit_available() or Chem is None:
            return self.subgraph_fail_reward

        parent = parse_smiles(
            original_smiles,
            sanitize=True,
            canonicalize=False,
            allow_capped_fragments=False,
        )
        fragment = parse_smiles(
            generated_smiles,
            sanitize=True,
            canonicalize=False,
            allow_capped_fragments=True,
        )

        if not parent.sanitized or parent.mol is None:
            return self.subgraph_fail_reward
        if not fragment.sanitized or fragment.mol is None:
            return self.subgraph_fail_reward
        if not is_connected_fragment(generated_smiles):
            return self.subgraph_fail_reward

        # 带 `*` 的情况必须走 query 逻辑，否则 dummy atom 会被当成普通原子，
        # 无法表达“边界可匹配任意父分子相邻原子”的语义。
        if fragment.contains_dummy_atoms:
            query = self._build_capped_query(generated_smiles)
            if query is None:
                return self.subgraph_fail_reward

            if not parent.mol.HasSubstructMatch(query, useChirality=True):
                return self.subgraph_fail_reward

            if is_valid_capped_subgraph(original_smiles, generated_smiles):
                return self.subgraph_pass_reward
            return self.subgraph_fail_reward

        if parent.mol.HasSubstructMatch(fragment.mol, useChirality=True):
            return self.subgraph_pass_reward
        return self.subgraph_fail_reward

    @staticmethod
    def _build_capped_query(generated_smiles: str) -> object | None:
        """把带 `*` 的 SMILES 转成 SMARTS query。

        这里故意不用 `MolFromSmiles` 做 query，因为我们需要 `*`
        以 wildcard/query atom 的语义参与子结构匹配。
        """

        if Chem is None:
            return None

        try:
            return Chem.MolFromSmarts(generated_smiles)
        except Exception:  # pragma: no cover - 取决于 RDKit 内部行为
            return None
