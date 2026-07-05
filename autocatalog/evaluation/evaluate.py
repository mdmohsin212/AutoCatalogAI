import time
import numpy as np
import torch
from tqdm.auto import tqdm
from autocatalog.evaluation.metrics import collect_predictions, evaluate_predictions

def build_consistency_rule(dataframe, source_task, target_task):
    rules = {}
    for source_value, group in dataframe.groupby(source_task):
        counts = group[target_task].value_counts()
        rules[source_value] = {
            "target": counts.index[0],
            "dominance": float(counts.iloc[0] / counts.sum()),
        }

    return rules


def build_consistency_rules(train_df):
    return {
        "article_to_master": build_consistency_rule(
            train_df,
            "articleType",
            "masterCategory",
        ),
        "article_to_sub": build_consistency_rule(
            train_df,
            "articleType",
            "subCategory",
        ),
        "article_to_usage": build_consistency_rule(
            train_df,
            "articleType",
            "usage",
        ),
        "article_to_season": build_consistency_rule(
            train_df,
            "articleType",
            "season",
        ),
    }


def apply_consistency_rules(y_pred, y_probs, label_maps, rules):
    corrected = {
        task: predictions.copy()
        for task, predictions in y_pred.items()
    }

    mappings = [
        ("article_to_master", "masterCategory", 0.95),
        ("article_to_sub", "subCategory", 0.90),
        ("article_to_usage", "usage", 0.92),
        ("article_to_season", "season", 0.92),
    ]

    for index, article_id in enumerate(corrected["articleType"]):
        article_id = int(article_id)
        confidence = float(
            y_probs["articleType"][index, article_id]
        )

        if confidence < 0.65:
            continue

        article_label = label_maps["articleType"]["id2label"][str(article_id)]
        for rule_name, target_task, minimum_dominance in mappings:
            rule = rules[rule_name].get(article_label)
            if not rule:
                continue
            if rule["dominance"] < minimum_dominance:
                continue
            corrected[target_task][index] = label_maps[target_task]["label2id"][
                rule["target"]
            ]

    return corrected


def evaluate_loader(model, loader, device, tasks, label_maps, rules):
    y_true, y_pred, y_probs, indices = collect_predictions(model, loader, device, tasks)
    raw_metrics = evaluate_predictions(y_true, y_pred, y_probs, tasks)

    corrected_predictions = apply_consistency_rules(y_pred, y_probs, label_maps, rules)
    corrected_metrics = evaluate_predictions(y_true, corrected_predictions, y_probs, tasks)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "indices": indices,
        "corrected_pred": corrected_predictions,
        "raw_metrics": raw_metrics,
        "corrected_metrics": corrected_metrics,
    }


@torch.inference_mode()
def benchmark_single_image_latency(
    model,
    dataset,
    device,
    warmup_runs=20,
    measured_runs=100,
):
    model.eval()
    sample = dataset[0]

    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    color_features = sample["color_features"].unsqueeze(0).to(device)
    for _ in range(warmup_runs):
        model(pixel_values, color_features)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(measured_runs):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()
        model(pixel_values, color_features)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "average_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "runs": measured_runs,
    }


@torch.inference_mode()
def benchmark_batch_latency(
    model,
    loader,
    device,
    max_batches=100,
):
    model.eval()
    times = []
    for batch_index, batch in enumerate(
        tqdm(
            loader,
            desc="Batch benchmark",
            leave=False,
        )
    ):
        if batch_index >= max_batches:
            break

        pixel_values = batch["pixel_values"].to(device)
        color_features = batch["color_features"].to(device)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()
        model(pixel_values, color_features)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed / pixel_values.size(0))

    return {
        "average_ms_per_image": float(np.mean(times)),
        "p50_ms_per_image": float(np.percentile(times, 50)),
        "p95_ms_per_image": float(np.percentile(times, 95)),
        "batches": len(times),
    }