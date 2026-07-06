| Model | Avg Accuracy | Avg Macro F1 | Top-3 Accuracy | Exact Match | Latency |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 42.63% | 7.93% | 73.44% | 0.95% | 0.001 ms |
| Frozen CLIP + Heads (V1) | 83.35% | 65.68% | 97.11% | 27.94% | 6.227 ms |
| AutoCatalogAI V2 | 87.48% | 67.41% | 98.15% | 40.46% | 7.474 ms |

> All models were evaluated on the same 6,611-image held-out test split. Metrics use raw model predictions. Latency is batch-size-1 model-forward time with preprocessing excluded.