# Multi-seed campaign status (phase-0 hard completion gate)

*Generated 2026-09-25T13:00:33+00:00 from disk state. A checkpoint alone is NOT completion evidence; 'COMPLETE' requires history + val metrics + test metrics.*

| Seed | Status | Epochs run | Best Val AUC (epoch) | Best Checkpoint | last_state | Test Done |
| --- | --- | ---: | --- | --- | --- | --- |
| 1 | RUNNING | 9 | 0.9000 (7) | yes | yes | NO |
| 2 | PENDING | None | n/a | NO | NO | NO |
| 3 | PENDING | None | n/a | NO | NO | NO |
| 4 | PENDING | None | n/a | NO | NO | NO |
| 4 repeat | PENDING | None | n/a | NO | NO | NO |

**ALL_RUNS_COMPLETE: NO**

Per the hard training-completion gate, final model selection, reliability gates and product finalization wait until ALL_RUNS_COMPLETE: YES.