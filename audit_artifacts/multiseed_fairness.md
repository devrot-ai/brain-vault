# BrainVuln Multi-Seed Fairness Check

Checkpoint-embedded TrainConfig compared across runs.
Recipe fields compared: lr, weight_decay, batch_size, max_epochs, patience, warmup_epochs, cosine_epochs, pos_weight_auto, normalization

| Run key (dir seed) | Embedded seed |
| --- | --- |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 42 | 42 |
| 44 | 4 |

Recipe mismatches vs run 42: NONE — only the seed differs

Test subject set identical across 6 evaluated runs: True

Note: normalization/augmentation flags are not embedded in the checkpoint; the driver (scripts/multiseed.bat) passes the identical canonical recipe to every seed (--epochs 40 --patience 8 --warmup 2 --cos-epochs 12 --aug-strength light). The interrupted seed-3 run was restarted with the same recipe, so its history is a fresh deterministic replay of the same seed.
