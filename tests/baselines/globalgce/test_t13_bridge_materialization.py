from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from src.baselines import globalgce_frozen_gine_bridge as optimized
from src.baselines.t13_bridge_materialization_benchmark import (
    _equal, _rng, bridge, compare_records, fixture, model, reference_module, training_arm,
)


@pytest.fixture(scope="module")
def original():
    torch.set_num_threads(1)
    return reference_module(Path(__file__).resolve().parents[3])[0]


@pytest.mark.parametrize("target", [0, 2])
@pytest.mark.parametrize("options", [dict(nodes=1), dict(nodes=7, hole=True),
                                      dict(nodes=9, ring=True), dict(nodes=7, invalid=True)])
def test_immutable_baseline_loss_grad_update_rng_reload(original, options, target):
    frozen = model()
    old, new = bridge(original, frozen), bridge(optimized, frozen)
    values = fixture(**options)
    initial = [v.clone() for v in values]
    before = _rng()
    reference = training_arm(old, values, target)
    result = training_arm(new, values, target)
    reloaded = training_arm(new, values, target, reload_after_one=True)
    assert compare_records(reference, result)["exact"]
    assert compare_records(result, reloaded)["exact"]
    assert _equal(before, _rng())
    assert _equal(initial, values)
    assert all(p.grad is None and not p.requires_grad for p in new.parameters())


def test_directed_edge_order_and_hard_fields(original):
    values = fixture(nodes=6, hole=True, ring=True)
    kwargs = dict(features=values[0][0], adjacency=values[1][0], edge_attributes=values[2][0],
                  atom_symbols=("C", "O"), bond_names=("no_edge", "single", "double", "triple"),
                  schema=optimized.MolecularFeatureSchema.from_dict(model_schema()))
    old, new = original._hard_graph(**kwargs), optimized._hard_graph(**kwargs)
    for field in old.__dataclass_fields__:
        assert _equal(getattr(old, field), getattr(new, field))
    frozen = model()
    captures = []
    for module in (original, optimized):
        call = bridge(module, frozen)
        layer = call.model.layers[0]
        actual = layer._aggregate_sum
        def capture(messages, target_indices, size):
            captures.append((messages.detach().clone(), target_indices.detach().clone(), size))
            return actual(messages, target_indices, size)
        layer._aggregate_sum = capture
        call(*values)
    assert _equal(captures[0], captures[1])


def model_schema():
    from src.data.molecular_graph_featurizer import default_molecular_feature_schema
    return default_molecular_feature_schema().to_dict()


def test_asymmetric_first_failure_is_preserved(original):
    values = fixture(nodes=7, hole=True)
    values[1][0, 0, 2] = 0.0
    errors = []
    for module in (original, optimized):
        with pytest.raises(ValueError) as exc:
            bridge(module, model())(*values)
        errors.append(str(exc.value))
    assert errors[0] == errors[1] == "GlobalGCE hard adjacency is asymmetric at (0,2)"
