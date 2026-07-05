import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def top_k_accuracy(y_true, probabilities, k=3):
    k = min(k, probabilities.shape[1])
    top_indices = np.argsort(
        -probabilities,
        axis=1,
    )[:, :k]

    matches = [
        y_true[index] in top_indices[index]
        for index in range(len(y_true))
    ]

    return float(np.mean(matches))


@torch.inference_mode()
def collect_predictions(model, loader, device, tasks):
    model.eval()

    y_true = {task: [] for task in tasks}
    y_pred = {task: [] for task in tasks}
    y_probs = {task: [] for task in tasks}
    global_indices = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        color_features = batch["color_features"].to(device)
        outputs = model(pixel_values, color_features)

        for task in tasks:
            probabilities = torch.softmax(outputs[task], dim=1)
            predictions = probabilities.argmax(dim=1)

            y_true[task].extend(
                batch["labels"][task].numpy().tolist()
            )

            y_pred[task].extend(
                predictions.cpu().numpy().tolist()
            )

            y_probs[task].extend(
                probabilities.cpu().numpy().tolist()
            )

        global_indices.extend(batch["global_indices"])

    for task in tasks:
        y_true[task] = np.asarray(y_true[task], dtype=np.int64)
        y_pred[task] = np.asarray(y_pred[task], dtype=np.int64)
        y_probs[task] = np.asarray(y_probs[task], dtype=np.float32)

    return y_true, y_pred, y_probs, global_indices


def evaluate_predictions(y_true, y_pred, y_probs, tasks):
    task_metrics = {}
    
    for task in tasks:
        task_metrics[task] = {
            "accuracy": float(
                accuracy_score(y_true[task], y_pred[task])
            ),
            "macro_f1": float(
                f1_score(
                    y_true[task],
                    y_pred[task],
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_true[task],
                    y_pred[task],
                    average="weighted",
                    zero_division=0,
                )
            ),
            "top3_accuracy": top_k_accuracy(
                y_true[task],
                y_probs[task],
                k=3,
            ),
        }

    exact_matches = np.ones(
        len(y_true[tasks[0]]),
        dtype=bool,
    )

    for task in tasks:
        exact_matches &= y_true[task] == y_pred[task]

    overall_metrics = {
        "average_accuracy": float(
            np.mean(
                [
                    task_metrics[task]["accuracy"]
                    for task in tasks
                ]
            )
        ),
        "average_macro_f1": float(
            np.mean(
                [
                    task_metrics[task]["macro_f1"]
                    for task in tasks
                ]
            )
        ),
        "average_weighted_f1": float(
            np.mean(
                [
                    task_metrics[task]["weighted_f1"]
                    for task in tasks
                ]
            )
        ),
        "average_top3_accuracy": float(
            np.mean(
                [
                    task_metrics[task]["top3_accuracy"]
                    for task in tasks
                ]
            )
        ),
        "exact_match_accuracy": float(exact_matches.mean()),
        "samples": int(len(exact_matches)),
    }

    return {
        "task_metrics": task_metrics,
        "overall_metrics": overall_metrics,
    }


def model_selection_score(metrics):
    overall = metrics["overall_metrics"]
    color_accuracy = metrics["task_metrics"]["baseColour"]["accuracy"]

    return float(
        0.25 * overall["average_accuracy"]
        + 0.35 * overall["average_macro_f1"]
        + 0.20 * overall["exact_match_accuracy"]
        + 0.20 * color_accuracy
    )


def passes_safety_thresholds(metrics, thresholds):
    task_metrics = metrics["task_metrics"]
    overall = metrics["overall_metrics"]

    return bool(
        overall["average_accuracy"]
        >= thresholds["average_accuracy"]
        and task_metrics["masterCategory"]["accuracy"]
        >= thresholds["masterCategory_accuracy"]
        and task_metrics["subCategory"]["accuracy"]
        >= thresholds["subCategory_accuracy"]
        and task_metrics["articleType"]["accuracy"]
        >= thresholds["articleType_accuracy"]
        and task_metrics["usage"]["accuracy"]
        >= thresholds["usage_accuracy"]
    )