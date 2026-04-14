"""统一的 PPO 奖励包装器。

该模块实现三件事：

1. 加载轻量级 Oracle（RandomForest）；
2. 把 SMILES 转成 Morgan 指纹；
3. 组合 `R_valid + R_subgraph + R_counterfactual`。

重要设计说明：
- 反事实奖励严格基于“删除生成子图后的 residual 分子”来计算；
- 不采用“片段单独预测原标签/对立标签”的定义，因为那会偏离仓库 v3 目标；
- 对于 residual 或片段中的 `*`，我们会先做 dummy atom 清理，再送入 Oracle。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from src.chem.deletion import delete_fragment_from_parent
from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule
from src.rewards.chem_rules import ChemRewardEngine

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:  # pragma: no cover - 兼容旧版 RDKit
        GetMorganGenerator = None
except ImportError:  # pragma: no cover - 取决于本地环境
    Chem = None
    DataStructs = None
    AllChem = None
    GetMorganGenerator = None


OracleBundle = dict[str, Any]


def load_oracle_bundle(model_path: str | Path) -> OracleBundle:
    """加载训练好的 Oracle bundle。

    bundle 采用纯 `dict` 持久化，避免把特征提取逻辑绑死在 `__main__`
    或脚本内部类上，保证训练脚本保存后，PPO 侧可以稳定反序列化。
    """

    bundle_path = Path(model_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Oracle model file does not exist: {bundle_path}")

    try:
        with bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading the Oracle bundle requires its runtime dependencies to be "
            "installed, for example scikit-learn."
        ) from exc

    _validate_oracle_bundle(bundle, bundle_path)
    return bundle


def _validate_oracle_bundle(bundle: Any, bundle_path: Path | None = None) -> None:
    """校验 pickle 文件里是否包含奖励模块需要的最小字段。"""

    if not isinstance(bundle, dict):
        source = f" from {bundle_path}" if bundle_path is not None else ""
        raise ValueError(f"Oracle bundle{source} must be a dictionary.")

    required_keys = ("model", "fingerprint_radius", "fingerprint_bits")
    missing_keys = [key for key in required_keys if key not in bundle]
    if missing_keys:
        source = f" from {bundle_path}" if bundle_path is not None else ""
        raise ValueError(
            f"Oracle bundle{source} is missing required keys: {missing_keys}"
        )

    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Oracle model must implement predict_proba(...).")


def _clear_non_ring_aromatic_flags(mol: object) -> None:
    """修复 dummy atom 删除后常见的芳香性残留问题。"""

    if Chem is None:
        return

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)

    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)


def prepare_smiles_for_oracle(smiles: str) -> str | None:
    """把带 `*` 的 SMILES 清理成适合送入 Oracle 的分子 SMILES。

    这里不直接做字符串级别的 `replace("*", "")`，因为：
    - `C(*)C -> C()C` 会变成非法字符串；
    - 芳香体系和支链结构需要通过 RDKit 图级删除 dummy atom 后再 sanitize。

    返回值约定：
    - 返回规范化后的可用 SMILES；
    - 若删除 dummy atom 后分子为空，返回空字符串 `""`；
    - 若无法修复，返回 `None`。
    """

    if not is_rdkit_available() or Chem is None:
        return None

    parsed = parse_smiles(
        smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not parsed.sanitized or parsed.mol is None:
        return None

    if not parsed.contains_dummy_atoms:
        return Chem.MolToSmiles(parsed.mol, canonical=True)

    rw_mol = Chem.RWMol(parsed.mol)
    dummy_atom_indices = sorted(
        (atom.GetIdx() for atom in parsed.mol.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    for atom_index in dummy_atom_indices:
        rw_mol.RemoveAtom(atom_index)

    repaired = rw_mol.GetMol()
    if repaired.GetNumAtoms() == 0:
        return ""

    sanitized, _, _, _ = sanitize_molecule(
        repaired,
        allow_capped_fragments=False,
    )
    if sanitized is None:
        _clear_non_ring_aromatic_flags(repaired)
        sanitized, _, _, _ = sanitize_molecule(
            repaired,
            allow_capped_fragments=False,
        )
    if sanitized is None:
        return None

    return Chem.MolToSmiles(sanitized, canonical=True)


@lru_cache(maxsize=8)
def _get_morgan_generator(radius: int, n_bits: int) -> object | None:
    """缓存 Morgan 指纹生成器，减少 PPO 高频调用下的对象构建开销。"""

    if GetMorganGenerator is None:
        return None
    return GetMorganGenerator(radius=radius, fpSize=n_bits)


def smiles_to_morgan_array(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    clean_dummy_atoms: bool = True,
) -> np.ndarray | None:
    """把一个 SMILES 转成 Morgan 指纹的 numpy 向量。

    `clean_dummy_atoms=True` 时，会先把 `*` 号通过图编辑方式清理掉，
    以便 residual/capped fragment 也能被 Oracle 正常打分。
    """

    if not is_rdkit_available() or Chem is None or DataStructs is None:
        return None

    normalized_smiles = smiles
    if clean_dummy_atoms:
        normalized_smiles = prepare_smiles_for_oracle(smiles)
        if normalized_smiles is None:
            return None
        if normalized_smiles == "":
            return None

    parsed = parse_smiles(
        normalized_smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=False,
    )
    if not parsed.sanitized or parsed.mol is None:
        return None

    fp_array = np.zeros((n_bits,), dtype=np.float32)

    generator = _get_morgan_generator(radius, n_bits)
    if generator is not None:
        fingerprint = generator.GetFingerprint(parsed.mol)
    else:
        if AllChem is None:  # pragma: no cover - 理论上不会到这里
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            parsed.mol,
            radius,
            nBits=n_bits,
        )

    DataStructs.ConvertToNumpyArray(fingerprint, fp_array)
    return fp_array


@dataclass(slots=True)
class CounterfactualReward:
    """PPO 阶段可直接调用的统一奖励计算器。"""

    model_path: str | Path | None = Path("outputs/oracle/aids_rf_model.pkl")
    oracle_bundle: OracleBundle | None = None
    chem_engine: ChemRewardEngine = field(default_factory=ChemRewardEngine)
    valid_pass_reward: float = 1.0
    invalid_smiles_penalty: float = -2.0
    subgraph_pass_reward: float = 1.0
    invalid_subgraph_penalty: float = -2.0
    oracle_failure_penalty: float = -1.0
    empty_residual_penalty: float = -1.0
    flip_bonus: float = 1.0
    flip_threshold: float = 0.5
    model: Any = field(init=False, repr=False)
    fingerprint_radius: int = field(init=False)
    fingerprint_bits: int = field(init=False)
    class_labels: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bundle = self.oracle_bundle
        if bundle is None:
            if self.model_path is None:
                raise ValueError("Either model_path or oracle_bundle must be provided.")
            bundle = load_oracle_bundle(self.model_path)
        else:
            _validate_oracle_bundle(bundle)

        self.oracle_bundle = bundle
        self.model = bundle["model"]
        self.fingerprint_radius = int(bundle["fingerprint_radius"])
        self.fingerprint_bits = int(bundle["fingerprint_bits"])
        raw_labels = bundle.get("class_labels")
        if raw_labels is None:
            raw_labels = getattr(self.model, "classes_", (0, 1))
        self.class_labels = tuple(int(label) for label in raw_labels)

    def compute_reward(
        self,
        original_smiles: str,
        generated_smiles: str,
        original_activity: int,
    ) -> tuple[float, dict[str, float]]:
        """计算 PPO step 使用的总奖励与拆解项。

        逻辑顺序：
        1. 先算 `R_valid`，失败直接早停；
        2. 再算 `R_subgraph`，失败直接早停；
        3. 删除生成片段，构造 residual；
        4. residual 送入 Oracle，得到对立类别概率；
        5. 基于 flip 与概率 margin 计算 `R_counterfactual`。
        """

        original_label = int(original_activity)
        if original_label not in (0, 1):
            raise ValueError("original_activity must be 0 or 1.")

        valid_score = self.chem_engine.check_validity(generated_smiles)
        if valid_score < 0:
            breakdown = {
                "valid_r": self.invalid_smiles_penalty,
                "subgraph_r": 0.0,
                "cf_r": 0.0,
            }
            return float(sum(breakdown.values())), breakdown

        subgraph_score = self.chem_engine.check_subgraph(
            original_smiles=original_smiles,
            generated_smiles=generated_smiles,
        )
        if subgraph_score < 0:
            breakdown = {
                "valid_r": self.valid_pass_reward,
                "subgraph_r": self.invalid_subgraph_penalty,
                "cf_r": 0.0,
            }
            return float(sum(breakdown.values())), breakdown

        deletion_result = delete_fragment_from_parent(original_smiles, generated_smiles)
        if not deletion_result.success or deletion_result.residual_smiles is None:
            breakdown = {
                "valid_r": self.valid_pass_reward,
                "subgraph_r": self.subgraph_pass_reward,
                "cf_r": self.oracle_failure_penalty,
            }
            return float(sum(breakdown.values())), breakdown

        residual_smiles = prepare_smiles_for_oracle(deletion_result.residual_smiles)
        if residual_smiles is None:
            breakdown = {
                "valid_r": self.valid_pass_reward,
                "subgraph_r": self.subgraph_pass_reward,
                "cf_r": self.oracle_failure_penalty,
            }
            return float(sum(breakdown.values())), breakdown
        if residual_smiles == "":
            breakdown = {
                "valid_r": self.valid_pass_reward,
                "subgraph_r": self.subgraph_pass_reward,
                "cf_r": self.empty_residual_penalty,
            }
            return float(sum(breakdown.values())), breakdown

        fingerprint = smiles_to_morgan_array(
            residual_smiles,
            radius=self.fingerprint_radius,
            n_bits=self.fingerprint_bits,
            clean_dummy_atoms=False,
        )
        if fingerprint is None:
            breakdown = {
                "valid_r": self.valid_pass_reward,
                "subgraph_r": self.subgraph_pass_reward,
                "cf_r": self.oracle_failure_penalty,
            }
            return float(sum(breakdown.values())), breakdown

        class_probability = self._predict_class_probability(
            fingerprint,
            target_label=1 - original_label,
        )
        original_probability = self._predict_class_probability(
            fingerprint,
            target_label=original_label,
        )
        cf_r = class_probability - original_probability
        if class_probability >= self.flip_threshold:
            cf_r += self.flip_bonus

        breakdown = {
            "valid_r": self.valid_pass_reward,
            "subgraph_r": self.subgraph_pass_reward,
            "cf_r": float(cf_r),
        }
        return float(sum(breakdown.values())), breakdown

    def _predict_class_probability(
        self,
        fingerprint: np.ndarray,
        *,
        target_label: int,
    ) -> float:
        """从 `predict_proba` 输出里稳定取出目标类别概率。"""

        probabilities = self.model.predict_proba(fingerprint.reshape(1, -1))[0]
        probability_map = {
            int(label): float(probability)
            for label, probability in zip(self.class_labels, probabilities, strict=True)
        }
        return float(probability_map.get(int(target_label), 0.0))
