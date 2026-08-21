# TasteMolNet multiclass baseline audit

Date: 2026-08-22

This audit describes compatibility with a shared three-class frozen GNN. It is
not evidence that any TasteMolNet baseline has been run.

| Method | Status | Evidence and required work |
|---|---|---|
| Ours | EASY_UNTARGETED_EXTENSION | The selector already consumes verified effects, but the historical reward/oracle stack is binary/RF. Route every new stage through the generic GNN oracle and shared multiclass semantics. |
| GCFExplainer | EASY_UNTARGETED_EXTENSION | The upstream GNN primitive can emit multiple classes, but project BACE/Mutagenicity adapters fix two classes and a label-0 target. Inject the shared three-class oracle and remove adapter-specific target assumptions for the new route only. |
| GlobalGCE | EASY_UNTARGETED_EXTENSION | Its unified evaluator already defines strict flip as `pred_before == source_label` and `pred_after != source_label`; project dataset adapters and native training remain binary. Add a Taste-specific adapter without changing historical outputs. |
| COMRECGC | EASY_UNTARGETED_EXTENSION | The graph walk/action representation is label-agnostic, while the project model adapter and slot evaluation currently fix two classes and a destination label. Parameterize the new route around the shared oracle. |
| CLEAR | BINARY_HARDCODED | The model loader can infer a class count, but candidate selection reconstructs two RF probabilities and rejects target labels outside `{0,1}`. A dedicated multiclass conversion/evaluation change is required before a Taste run. |
| Other paper baselines | NOT_YET_AUDITED | A baseline enters TasteMolNet only after its action semantics and three-class oracle interface are proven. |

## Shared requirements

Future TasteMolNet baselines must:

- use the exact same frozen three-class checkpoint;
- count both Sweet-to-Bitter and Sweet-to-Tasteless as untargeted flips;
- record full probability vectors and destination distributions;
- fit selectors on calibration only;
- never convert the task to one-vs-rest;
- never use a method-specific RF oracle.
