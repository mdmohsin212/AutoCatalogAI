import json
import torch
from huggingface_hub import hf_hub_download

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def download_source_checkpoint(repo_id):
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
    )

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
    )

    labels_path = hf_hub_download(
        repo_id=repo_id,
        filename="label_maps.json",
    )

    source_config = load_json(config_path)
    label_maps = load_json(labels_path)
    checkpoint = torch_load(model_path)

    model_name = (
        source_config.get("base_model_name")
        or source_config.get("model_name")
        or checkpoint.get("model_name")
        or "openai/clip-vit-base-patch32"
    )

    hidden_dim = int(
        source_config.get(
            "hidden_dim",
            checkpoint.get("hidden_dim", 512),
        )
    )

    dropout = float(
        source_config.get(
            "dropout",
            checkpoint.get("dropout", 0.2),
        )
    )

    return checkpoint, label_maps, model_name, hidden_dim, dropout


def save_checkpoint(
    path,
    model,
    model_name,
    tasks,
    task_num_classes,
    hidden_dim,
    dropout,
    color_feature_dim,
    label_maps,
    epoch_name,
    validation_score,
    validation_metrics,
    source_repo_id,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "architecture": "CLIPMultiTaskClassifierV2",
        "architecture_version": "2.0",
        "tasks": tasks,
        "task_num_classes": task_num_classes,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "color_feature_dim": color_feature_dim,
        "label_maps": label_maps,
        "best_epoch": epoch_name,
        "best_validation_score": float(validation_score),
        "best_validation_metrics": validation_metrics,
        "source_v1_repo": source_repo_id,
    }

    torch.save(checkpoint, path)


def save_final_metadata(
    config,
    output_dir,
    label_maps,
    task_num_classes,
    model_name,
    hidden_dim,
    dropout,
    best_checkpoint,
    final_metrics,
):
    data_config = config["data"]
    training_config = config["training"]

    saved_config = {
        "project_name": config["project"]["name"],
        "version": config["project"]["version"],
        "dataset_name": data_config["dataset_name"],
        "source_v1_repo": config["model"]["source_repo_id"],
        "base_model_name": model_name,
        "architecture": "CLIPMultiTaskClassifierV2",
        "tasks": data_config["tasks"],
        "task_num_classes": task_num_classes,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "color_feature_dim": data_config["color_feature_dim"],
        "color_image_size": data_config["color_image_size"],
        "train_ratio": data_config["train_ratio"],
        "validation_ratio": data_config["validation_ratio"],
        "test_ratio": data_config["test_ratio"],
        "balanced_tasks": training_config["balanced_tasks"],
        "class_weight_min": training_config["class_weight_min"],
        "class_weight_max": training_config["class_weight_max"],
        "best_checkpoint": best_checkpoint["best_epoch"],
        "best_validation_score": best_checkpoint["best_validation_score"],
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(
            saved_config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(output_dir / "label_maps.json", "w", encoding="utf-8") as file:
        json.dump(
            label_maps,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            final_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )