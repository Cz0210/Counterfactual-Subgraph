"""CPU fixture only: protect the pinned five-batch update and loader boundary."""
import copy
from pathlib import Path

import pytest
import torch

from src.baselines.t13_component_diagnostics import cpu_copy, exact_difference
from src.baselines.t13_real_batch_performance import (
    ObservedOracle, run_formal_diagnostic_update,
)


class Loader:
    def __init__(self, count, events, prefix):
        self.count, self.events, self.prefix = count, events, prefix
        self.dataset = range(count * 2)
        self.source = [dict(index=torch.tensor([2*i, 2*i+1]),
                            value=torch.tensor([.2 + i*.1, .3 + i*.1])) for i in range(count)]

    def __len__(self):
        return self.count

    def __iter__(self):
        self.events.append(self.prefix + ':iter')
        torch.rand(())  # Same timing-sensitive global RNG use as loader iteration.
        for index, value in enumerate(self.source):
            self.events.append(self.prefix + ':fetch' + str(index))
            yield copy.deepcopy(value)


class Oracle:
    def eval(self):
        return self


class Model(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.4, .7]))
        self.gt_gnn = Oracle()
        self.events = events
        self.weight.register_hook(lambda grad: self.events.append('backward'))

    def get_rules(self, fss):
        self.events.append('rules')
        return self.weight.square() + torch.rand(2) * .1

    def run_one_batch(self, rules, batch):
        self.events.append('train' if self.training else 'validation')
        batch['value'].add_(.05)  # Native mutable-batch behavior must not leak.
        value = rules * batch['value']
        if self.training:
            value = torch.nn.functional.dropout(value, .2, training=True)
        loss = value.square().sum()
        return loss, value.sum() * .1, value.square().sum() * .2, (value - .8).square().sum()


def validation(loader, model, pred, rules):
    pred.eval()
    with torch.no_grad():
        loss = kl = sim = cfe = 0.0
        count = 0
        for _, data in enumerate(loader):
            count += 1
            values = model.run_one_batch(rules, data)
            loss, kl, sim, cfe = loss+values[0], kl+values[1], sim+values[2], cfe+values[3]
        return dict(loss=loss/count, loss_kl=kl/count, loss_sim=sim/count, loss_cfe=cfe/count)


def setup():
    events = []
    model = Model(events)
    class Adam(torch.optim.Adam):
        def step(self, *args, **kwargs):
            events.append('optimizer.step')
            return super().step(*args, **kwargs)
        def zero_grad(self, *args, **kwargs):
            events.append('optimizer.zero_grad')
            return super().zero_grad(*args, **kwargs)
    class Scheduler(torch.optim.lr_scheduler.StepLR):
        def step(self, *args, **kwargs):
            events.append('scheduler.step')
            return super().step(*args, **kwargs)
    optimizer = Adam(model.parameters(), lr=.1, weight_decay=1e-5)
    scheduler = Scheduler(optimizer, step_size=10, gamma=.9)
    events.clear()
    return model, optimizer, scheduler, Loader(7, events, 'train'), Loader(3, events, 'val'), events


def original_update(model, train, val, optimizer, scheduler, epoch, best_loss):
    # Verbatim arithmetic/control order from c0eb892d globalgce_resumable.py.
    model.train()
    model.gt_gnn.eval()
    loss = loss_kl = loss_sim = loss_cfe = 0.0
    rules = model.get_rules(None)
    for batch_index, data in enumerate(train):
        if batch_index >= 5:
            break
        values = model.run_one_batch(rules, data)
        loss += values[0]
        loss_kl += values[1]
        loss_sim += values[2]
        loss_cfe += values[3]
    (loss_cfe if epoch < 35 else loss).backward()
    gradients = {name: cpu_copy(value.grad) for name, value in model.named_parameters()}
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    evaluated = None
    if epoch % 5 == 0:
        with torch.no_grad():
            model.eval()
            evaluated = validation(val, model, model.gt_gnn, rules)
            val_loss = float(evaluated['loss'].detach().cpu())
            if val_loss < best_loss:
                best_loss = val_loss
    return dict(losses=cpu_copy((loss, loss_kl, loss_sim, loss_cfe)), gradients=gradients,
                validation=cpu_copy(evaluated), best_loss=best_loss)


def state(model, optimizer, scheduler):
    return cpu_copy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), rng=torch.get_rng_state()))


@pytest.mark.parametrize('first_epoch', [30, 35])
def test_two_accumulated_updates_match_original_loss_gradient_update_rng_and_due_full_validation(first_epoch):
    torch.manual_seed(123)
    initial_rng = torch.get_rng_state().clone()
    native = setup()
    reference_rows = []
    best_loss = float('inf')
    for epoch in (first_epoch, first_epoch + 1):
        row = original_update(native[0], native[3], native[4], native[1], native[2], epoch, best_loss)
        best_loss = row['best_loss']
        row['after'] = state(*native[:3])
        reference_rows.append(row)
    torch.set_rng_state(initial_rng)
    repaired = setup()
    observed_steps = []
    best_loss = float('inf')
    for index, epoch in enumerate((first_epoch, first_epoch + 1)):
        row = run_formal_diagnostic_update(model=repaired[0], fss=None, train_loader=repaired[3],
            validation_loader=repaired[4], pred_model=repaired[0].gt_gnn, epoch=epoch,
            optimizer=repaired[1], scheduler=repaired[2], test_globalgce=validation,
            best_loss=best_loss, best_state_seen=index > 0,
            optimizer_step_observer=lambda: observed_steps.append('actual-step'))
        best_loss = row['best_loss']
        compare = {key: row[key] for key in ('losses', 'gradients', 'validation', 'best_loss')}
        compare['after'] = state(*repaired[:3])
        comparison = exact_difference(reference_rows[index], compare)
        assert comparison['exact'], comparison
        assert len(row['train_batch_bindings']) == 5
        if index == 0:
            assert row['validation_binding']['example_count'] == 6
            assert row['validation_binding']['batch_count'] == 3
            assert row['validation_binding']['complete'] is True
        else:
            assert row['validation'] is None and row['validation_binding'] is None
    assert native[-1] == repaired[-1]
    assert repaired[-1].count('rules') == 2
    assert repaired[-1].count('backward') == 2
    assert repaired[-1].count('train') == 10
    assert repaired[-1].count('train:fetch5') == 2  # Preserve original sixth fetch.
    assert repaired[-1].count('train:fetch6') == 0
    assert len(observed_steps) == 2
    assert repaired[-1][:2] == ['rules', 'train:iter']


def test_short_validation_evaluator_is_rejected_after_real_update_not_marked_pass():
    model, opt, sched, train, val, _ = setup()
    def incorrectly_truncated(loader, model, pred, rules):
        return validation([next(iter(loader))], model, pred, rules)
    with pytest.raises(ValueError, match='FULL_VALIDATION_WAS_NOT_FULLY_CONSUMED'):
        run_formal_diagnostic_update(model=model, fss=None, train_loader=train, validation_loader=val,
            pred_model=model.gt_gnn, epoch=30, optimizer=opt, scheduler=sched,
            test_globalgce=incorrectly_truncated, best_loss=float('inf'), best_state_seen=False)
    assert all(float(value['step']) == 1 for value in opt.state.values())


def test_fresh_reload_after_due_full_validation_matches_next_five_batch_update(tmp_path):
    torch.manual_seed(47)
    model, opt, sched, train, val, _ = setup()
    def update(parts, epoch, best_loss, best_seen):
        return run_formal_diagnostic_update(model=parts[0], fss=None, train_loader=parts[3],
            validation_loader=parts[4], pred_model=parts[0].gt_gnn, epoch=epoch,
            optimizer=parts[1], scheduler=parts[2], test_globalgce=validation,
            best_loss=best_loss, best_state_seen=best_seen)
    continuous = (model, opt, sched, train, val)
    row30 = update(continuous, 30, float('inf'), False)
    saved = dict(state(model, opt, sched), next_epoch=31,
                 best_loss=row30['best_loss'], best_state_seen=row30['best_state_seen'])
    path = tmp_path / 'diagnostic-not-promotable.pt'
    torch.save(saved, path)
    row31 = update(continuous, 31, saved['best_loss'], saved['best_state_seen'])
    expected = dict(row=row31, after=state(model, opt, sched))
    restored = torch.load(path, map_location='cpu', weights_only=False)
    assert exact_difference(saved, restored)['exact']
    fresh = setup()
    assert fresh[0] is not model and fresh[1] is not opt
    fresh[0].load_state_dict(restored['model'])
    fresh[1].load_state_dict(restored['optimizer'])
    fresh[2].load_state_dict(restored['scheduler'])
    torch.set_rng_state(restored['rng'])
    observed_row = update(fresh, restored['next_epoch'], restored['best_loss'], restored['best_state_seen'])
    observed = dict(row=observed_row, after=state(*fresh[:3]))
    comparison = exact_difference(expected, observed)
    assert comparison['exact'], comparison
    assert fresh[-1].count('train') == 5 and fresh[-1].count('validation') == 0


def test_raw_validation_outputs_are_all_observed_in_bounded_digest_without_extra_calls():
    class CallableOracle:
        calls = 0
        def __call__(self, value):
            self.calls += 1
            return dict(logits=torch.tensor([value, -value]), y_pred=torch.tensor(value > 0))
    oracle = CallableOracle()
    observed = ObservedOracle(oracle, cpu_copy)
    observed.begin_phase(compact=True)
    for value in range(1000):
        observed(value)
    summary = observed.phase_records()
    assert oracle.calls == summary['count'] == 1000
    assert observed.records == []
    assert summary['raw_records_retained'] is False
    original = summary['ordered_state_sha256']
    observed.begin_phase(compact=True)
    for value in reversed(range(1000)):
        observed(value)
    assert observed.phase_records()['ordered_state_sha256'] != original


def test_canary_never_substitutes_old_two_batch_one_val_or_retains_autograd_graph():
    source = (Path(__file__).parents[3] / 'src/baselines/t13_real_batch_performance.py').read_text()
    assert 'train_batches=5' in source and 'validation_batches="ALL_WHEN_DUE"' in source
    assert 'retain_graph=True' not in source
    assert 'next(iter(validation))' not in source
    assert 'range(UPDATES_PER_ARM)' in source and 'MAX_DIAGNOSTIC_UPDATES = 8' in source
