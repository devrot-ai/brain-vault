# BrainVuln Multi-Seed Checkpoint Manifest

| Run | Checkpoint | Size (MB) | SHA256 | Embedded seed | Embedded val AUC | Loads | First-layer shape |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| 42 (canonical) | results\ml\resnet_seed42\checkpoints\best.pt | 127.4 | `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9` **(canonical — must equal frozen SHA)** | 42 | 0.8 | yes | (64, 1, 7, 7, 7) |
| 1 | results\ml\resnet_seed1\checkpoints\best.pt | 127.4 | `69d918d1dc75a6cb…` | 1 | 0.9 | yes | (64, 1, 7, 7, 7) |
| 2 | results\ml\resnet_seed2\checkpoints\best.pt | 127.4 | `1ac69639777f7bd7…` | 2 | 0.8444444444444444 | yes | (64, 1, 7, 7, 7) |
| 3 | results\ml\resnet_seed3\checkpoints\best.pt | 127.4 | `f984a433556715f2…` | 3 | 0.85 | yes | (64, 1, 7, 7, 7) |
| 4 | results\ml\resnet_seed4\checkpoints\best.pt | 127.4 | `5781f47fb6881a3b…` | 4 | 0.9166666666666667 | yes | (64, 1, 7, 7, 7) |
| 4 repeat | results\ml\resnet_seed4_repeat\checkpoints\best.pt | 127.4 | `134287e02ed80e35…` | 4 | 0.9166666666666667 | yes | (64, 1, 7, 7, 7) |

Canonical seed-42 SHA256: `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`
