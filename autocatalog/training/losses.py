import numpy as np
import torch
import torch.nn as nn


def build_criterions(train_df, tasks, label_maps, balanced_tasks, minimum_weight, maximum_weight, device):
    criterions = {}
    summary = {}

    for task in tasks:
        if task not in balanced_tasks:
            criterions[task] = nn.CrossEntropyLoss()
            summary[task] = None
            continue

        label2id = label_maps[task]["label2id"]
        encoded = train_df[task].map(label2id).to_numpy(dtype=np.int64)
        
        counts = np.bincount(
            encoded,
            minlength=len(label2id),
        ).astype(np.float64)

        weights = np.sqrt(
            counts.sum() / (len(label2id) * np.maximum(counts, 1.0))
        )

        weights /= max(weights.mean(), 1e-8)
        weights = np.clip(
            weights,
            minimum_weight,
            maximum_weight,
        )

        weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        criterions[task] = nn.CrossEntropyLoss(weight=weights)
        summary[task] = {
            "min": float(weights.min().item()),
            "max": float(weights.max().item()),
            "mean": float(weights.mean().item()),
        }

    return criterions, summary


def compute_multitask_loss(outputs, labels, criterions, active_tasks,):
    losses = [
        criterions[task](outputs[task], labels[task])
        for task in active_tasks
    ]

    return torch.stack(losses).mean()