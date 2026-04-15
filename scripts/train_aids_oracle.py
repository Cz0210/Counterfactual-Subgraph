"""训练一个用于 PPO 奖励打分的轻量级 AIDS 活性 Oracle。

输入：
- `data/aids_dataset.csv`
- 需要包含 `smiles` 列，以及以下标签列之一：
  - `HIV_active`：MoleculeNet 风格的二分类标签，优先使用
  - `activity`：NCI 原始活性标签，支持 `CI` / `CM` / `CA`

输出：
- `outputs/oracle/aids_rf_model.pkl`

这个脚本采用：
- RDKit Morgan Fingerprint（radius=2, 2048 bits）
- RandomForestClassifier

保存内容是一个纯字典 bundle，便于 `src.rewards.reward_calculator`
在 PPO 阶段直接加载并调用。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np

from src.chem import is_rdkit_available
from src.rewards.reward_calculator import smiles_to_morgan_array

try:
    import pandas as pd
except ImportError:  # pragma: no cover - 取决于本地环境
    pd = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - 取决于本地环境
    RandomForestClassifier = None
    accuracy_score = None
    roc_auc_score = None
    train_test_split = None


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="Train a lightweight AIDS oracle.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/aids_dataset.csv"),
        help="输入 CSV 路径，默认是 data/aids_dataset.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/oracle/aids_rf_model.pkl"),
        help="模型输出路径，默认是 outputs/oracle/aids_rf_model.pkl",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit 数",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="RandomForest 树数量",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="RandomForest 最大深度，默认不限制",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="RandomForest 并行线程数",
    )
    return parser


def _require_training_dependencies() -> None:
    """在脚本入口统一检查依赖，给出更友好的错误信息。"""

    missing_deps: list[str] = []
    if not is_rdkit_available():
        missing_deps.append("rdkit")
    if pd is None:
        missing_deps.append("pandas")
    if (
        RandomForestClassifier is None
        or accuracy_score is None
        or roc_auc_score is None
        or train_test_split is None
    ):
        missing_deps.append("scikit-learn")

    if missing_deps:
        raise RuntimeError(
            "scripts/train_aids_oracle.py requires the following packages: "
            + ", ".join(missing_deps)
            + "."
        )


def _load_dataset(csv_path: Path) -> "pd.DataFrame":
    """读取并校验输入数据。"""

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "smiles" not in frame.columns:
        raise ValueError("Dataset CSV is missing required column: 'smiles'")

    if "HIV_active" in frame.columns:
        label_column = "HIV_active"
    elif "activity" in frame.columns:
        label_column = "activity"
    else:
        raise ValueError(
            "Dataset CSV must contain either 'HIV_active' or 'activity' as the label column."
        )

    frame = frame.loc[:, ["smiles", label_column]].copy()
    frame = frame.dropna(subset=["smiles", label_column])
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    raw_labels = frame[label_column]

    if label_column == "HIV_active":
        # MoleculeNet 版本已经是二分类标签，直接安全转换即可。
        frame["activity"] = pd.to_numeric(raw_labels, errors="coerce")
    else:
        # NCI 原始 HIV 数据常见的是字符串标签，需要先映射到 0/1。
        normalized_labels = raw_labels.astype(str).str.strip().str.upper()
        mapped_labels = normalized_labels.map({"CI": 0, "CM": 1, "CA": 1})
        numeric_labels = pd.to_numeric(normalized_labels, errors="coerce")
        frame["activity"] = mapped_labels.where(mapped_labels.notna(), numeric_labels)

    frame = frame.dropna(subset=["activity"])
    frame["activity"] = frame["activity"].astype(int)
    frame = frame[frame["activity"].isin((0, 1))]
    frame = frame[frame["smiles"] != ""]
    if frame.empty:
        raise ValueError("No usable rows remained after basic CSV filtering.")

    return frame.loc[:, ["smiles", "activity"]].reset_index(drop=True)


def _build_feature_matrix(
    frame: "pd.DataFrame",
    *,
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """把 CSV 中的 SMILES 批量转成 Morgan 指纹矩阵。"""

    features: list[np.ndarray] = []
    labels: list[int] = []
    invalid_smiles_count = 0

    for row in frame.itertuples(index=False):
        fp_array = smiles_to_morgan_array(
            row.smiles,
            radius=radius,
            n_bits=n_bits,
            clean_dummy_atoms=False,
        )
        if fp_array is None:
            invalid_smiles_count += 1
            continue
        features.append(fp_array)
        labels.append(int(row.activity))

    if not features:
        raise ValueError("No valid RDKit-parsable molecules remained after featurization.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if np.unique(y).size < 2:
        raise ValueError("Oracle training requires at least two classes after filtering.")

    return x, y, invalid_smiles_count


def _evaluate_classifier(
    model: "RandomForestClassifier",
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    """计算简单的 Accuracy / AUC。"""

    predicted_label = model.predict(x_eval)
    predicted_probability = model.predict_proba(x_eval)

    class_to_index = {int(label): idx for idx, label in enumerate(model.classes_)}
    positive_index = class_to_index.get(1)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_eval, predicted_label)),
    }
    if positive_index is not None and len(np.unique(y_eval)) > 1:
        metrics["auc"] = float(
            roc_auc_score(y_eval, predicted_probability[:, positive_index])
        )
    else:
        metrics["auc"] = float("nan")
    return metrics


def train_oracle(args: argparse.Namespace) -> Path:
    """训练模型并把 bundle 持久化到磁盘。"""

    _require_training_dependencies()

    frame = _load_dataset(args.data_path)
    x, y, invalid_smiles_count = _build_feature_matrix(
        frame,
        radius=args.radius,
        n_bits=args.n_bits,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    train_metrics = _evaluate_classifier(model, x_train, y_train)
    test_metrics = _evaluate_classifier(model, x_test, y_test)

    bundle = {
        "model": model,
        "task_name": "aids_activity",
        "feature_type": "rdkit_morgan",
        "fingerprint_radius": int(args.radius),
        "fingerprint_bits": int(args.n_bits),
        "positive_label": 1,
        "class_labels": [int(label) for label in model.classes_],
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "total_valid_examples": int(x.shape[0]),
        "invalid_smiles_filtered": int(invalid_smiles_count),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "random_seed": int(args.random_seed),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    print("Oracle training finished.")
    print(f"Saved bundle to: {args.output_path}")
    print(f"Valid examples: {x.shape[0]}")
    print(f"Filtered invalid SMILES: {invalid_smiles_count}")
    print(
        "Train metrics: "
        f"accuracy={train_metrics['accuracy']:.4f}, auc={train_metrics['auc']:.4f}"
    )
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, auc={test_metrics['auc']:.4f}"
    )
    return args.output_path


def main() -> None:
    """命令行入口。"""

    parser = build_argparser()
    args = parser.parse_args()
    train_oracle(args)


if __name__ == "__main__":
    main()
