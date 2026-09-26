# BrainVuln Multi-Seed Training Health

Generated from train_history.json + checkpoint files (read-only).

| Run | Epochs | Best Val AUC (epoch) | Early stop | NaN | Explode | Val collapse | Checkpoint |
| --- | -----: | --- | --- | --- | --- | --- | --- |
| 42 (canonical) | 20 | 0.8 (epoch 19) | yes (0 after best) | NO | no | no | loads (127 tensors, seed 42) |
| 1 | 16 | 0.9 (epoch 7) | yes (8 after best) | NO | no | no | loads (127 tensors, seed 1) |
| 2 | 13 | 0.8444 (epoch 4) | yes (8 after best) | NO | no | no | loads (127 tensors, seed 2) |
| 3 | 13 | 0.85 (epoch 4) | yes (8 after best) | NO | no | no | loads (127 tensors, seed 3) |
| 4 | 18 | 0.9167 (epoch 9) | yes (8 after best) | NO | no | no | loads (127 tensors, seed 4) |
| 4 repeat (reproducibility) | 18 | 0.9167 (epoch 9) | yes (8 after best) | NO | no | no | loads (127 tensors, seed 4) |

## Per-run detail

- **42 (canonical)** — {"run": "42 (canonical)", "dir": "results\\ml\\resnet_seed42", "epochs_run": 20, "nan_loss": false, "nonfinite": false, "exploding": false, "val_collapse": false, "best_val_auc": 0.8, "best_epoch": 19, "early_stop": true, "epochs_after_best": 0, "ckpt_loads": true, "n_tensors": 127, "ckpt_seed": 42, "ckpt_arch": "?", "ckpt_val_auc": 0.8, "status": "OK"}
- **1** — {"run": "1", "dir": "results\\ml\\resnet_seed1", "epochs_run": 16, "nan_loss": false, "nonfinite": false, "exploding": false, "val_collapse": false, "best_val_auc": 0.9, "best_epoch": 7, "early_stop": true, "epochs_after_best": 8, "ckpt_loads": true, "n_tensors": 127, "ckpt_seed": 1, "ckpt_arch": "?", "ckpt_val_auc": 0.9, "status": "OK"}
- **2** — {"run": "2", "dir": "results\\ml\\resnet_seed2", "epochs_run": 13, "nan_loss": false, "nonfinite": false, "exploding": false, "val_collapse": false, "best_val_auc": 0.8444, "best_epoch": 4, "early_stop": true, "epochs_after_best": 8, "ckpt_loads": true, "n_tensors": 127, "ckpt_seed": 2, "ckpt_arch": "?", "ckpt_val_auc": 0.8444444444444444, "status": "OK"}
- **3** — {"run": "3", "dir": "results\\ml\\resnet_seed3", "epochs_run": 13, "nan_loss": false, "nonfinite": false, "exploding": false, "val_collapse": false, "best_val_auc": 0.85, "best_epoch": 4, "early_stop": true, "epochs_after_best": 8, "ckpt_loads": true, "n_tensors": 127, "ckpt_seed": 3, "ckpt_arch": "?", "ckpt_val_auc": 0.85, "status": "OK"}
- **4** — {"run": "4", "dir": "results\\ml\\resnet_seed4", "epochs_run": 18, "nan_loss": false, "nonfinite": false, "exploding": false, "val_collapse": false, "best_val_auc": 0.9167, "best_epoch": 9, "early_stop": true, "epochs_after_best": 8, "ckpt_loads": true, "n_tensors": 127, "ckpt_seed": 4, "ckpt_arch": "?", "ckpt_val_auc": 0.9166666666666667, "status": "OK"}
- **4 repeat (reproducibility)** — {"run": "4 repeat (reproducibility)", "dir": "results\\ml\\resnet_seed4_repeat", "epochs_run": 18, "nan_loss": false, "nonfinite": false, "exploding": false, "val_collapse": false, "best_val_auc": 0.9167, "best_epoch": 9, "early_stop": true, "epochs_after_best": 8, "ckpt_loads": true, "n_tensors": 127, "ckpt_seed": 4, "ckpt_arch": "?", "ckpt_val_auc": 0.9166666666666667, "status": "OK"}

Overall: 6/6 runs healthy.
