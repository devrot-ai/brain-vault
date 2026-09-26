# Multi-seed campaign status (phase-0 hard completion gate)

*Generated 2026-09-26T15:10:00+00:00 from disk state. A checkpoint alone is NOT completion evidence; 'COMPLETE' requires history + val metrics + test metrics.*

| Seed | Status | Epochs run | Best Val AUC (epoch) | Best Checkpoint | last_state | Test Done |
| --- | --- | ---: | --- | --- | --- | --- |
| 1 | COMPLETE | 16 | 0.9000 (7) | yes | yes | yes |
| 2 | COMPLETE | 13 | 0.8444 (4) | yes | yes | yes |
| 3 | COMPLETE | 13 | 0.8500 (4) | yes | yes | yes |
| 4 | COMPLETE | 18 | 0.9167 (9) | yes | yes | yes |
| 4 repeat | COMPLETE | 18 | 0.9167 (9) | yes | yes | yes |

**ALL_RUNS_COMPLETE: YES**

Per the hard training-completion gate, final model selection, reliability gates and product finalization wait until ALL_RUNS_COMPLETE: YES.