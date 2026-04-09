# Config Layout

This directory stores versioned experiment configs for the counterfactual
subgraph v3 project.

The repository bootstrap phase only creates the folder skeleton:

- `configs/data/`
- `configs/model/`
- `configs/train/`
- `configs/reward/`
- `configs/eval/`

Concrete YAML or TOML files should be added after the chemistry, data, and
reward interfaces are stabilized.
