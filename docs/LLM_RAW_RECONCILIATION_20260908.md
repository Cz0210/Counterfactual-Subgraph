# LLM-GIN scoped raw-source correction

The old merger bound canonical graph pairs, weights and kernel source, but not
the actual historical embedding tensors and execution layout. Its conflicting
scalars cannot be adjudicated by closeness, minimum or source-method preference.
The new LLM-only opt-in preserves both observations, scans both splits once,
and recomputes only conflicted test pairs with the already-frozen CPU producer.
It retains actual attributed graphs, input tensors' identities, embedding/cost
arrays and solver inputs. The numerical producer is sealed before measurement.

No calibration conflict is silently waived: that case stops for an explicit
reduction/selector repair. With zero calibration conflicts, the original 264
parent records and global freeze are retained byte-for-byte. Reconciled test
parents bind the new index. Original GNN code/caches and historical results do
not change. New measurement does not prove a historical producer was correct.

Existing CLI: --action resume --raw-reconciliation-root PATH. The unchanged
Slurm CPU wrapper forwards it. There is no new scheduler, generation, fitting,
parser, matrix publisher, or GPU request. Final scientific acceptance and
transfer remain separate from EVALUATION_COMPLETE.
