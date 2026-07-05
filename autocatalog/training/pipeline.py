import json
import time
from pathlib import Path
import torch
from transformers import CLIPImageProcessor
from autocatalog.data.dataset import (
    build_dataloaders,
    create_splits,
    load_clean_dataset,
    load_or_create_color_cache,
)
from autocatalog.evaluation.error_analysis import save_evaluation_artifacts
from autocatalog.evaluation.evaluate import (
    benchmark_batch_latency,
    benchmark_single_image_latency,
    build_consistency_rules,
    evaluate_loader,
)
from autocatalog.evaluation.metrics import (
    model_selection_score,
    passes_safety_thresholds,
)
from autocatalog.models.multitask_clip import CLIPMultiTaskClassifierV2
from autocatalog.training.checkpoint import (
    download_source_checkpoint,
    save_checkpoint,
    save_final_metadata,
    torch_load,
)
from autocatalog.training.losses import build_criterions
from autocatalog.training.train import create_optimizer_and_scheduler, train_one_epoch
from autocatalog.utils.logger import get_logger
from autocatalog.utils.seed import set_seed
logger = get_logger(__name__)


def run_training(config, root_dir):
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    set_seed(data_config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = Path(root_dir)

    output_dir = root_dir / model_config["output_dir"]
    evaluation_dir = root_dir / evaluation_config["output_dir"]
    processed_dir = root_dir / data_config["processed_dir"]
    color_cache_path = root_dir / data_config["color_cache_path"]

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training started | device=%s | source=%s",
        device,
        model_config["source_repo_id"],
    )

    source_checkpoint, label_maps, model_name, hidden_dim, dropout = (
        download_source_checkpoint(model_config["source_repo_id"])
    )

    tasks = data_config["tasks"]
    task_num_classes = {
        task: len(label_maps[task]["label2id"])
        for task in tasks
    }

    dataset = load_clean_dataset(
        data_config["dataset_name"],
        tasks,
        label_maps,
    )

    train_df, validation_df, test_df = create_splits(
        dataset,
        tasks,
        processed_dir,
        data_config["seed"],
        data_config["train_ratio"],
        data_config["validation_ratio"],
        data_config["test_ratio"],
    )

    color_features = load_or_create_color_cache(
        dataset,
        color_cache_path,
        data_config["color_image_size"],
        data_config["color_feature_dim"],
    )

    processor = CLIPImageProcessor.from_pretrained(model_name)
    data = build_dataloaders(
        dataset,
        train_df,
        validation_df,
        test_df,
        color_features,
        processor,
        label_maps,
        tasks,
        training_config["batch_size"],
        training_config["num_workers"],
    )

    model = CLIPMultiTaskClassifierV2(
        model_name=model_name,
        task_num_classes=task_num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        color_feature_dim=data_config["color_feature_dim"],
    ).to(device)

    source_state = source_checkpoint.get("model_state_dict", source_checkpoint)
    load_result = model.load_state_dict(source_state, strict=False)
    expected_prefixes = (
        "master_to_sub",
        "sub_to_article",
        "article_to_season",
        "article_to_usage",
        "color_branch",
    )

    unexpected_missing = [
        key
        for key in load_result.missing_keys
        if not key.startswith(expected_prefixes)
    ]

    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError("Source checkpoint does not match the expected V1 model")

    logger.info("Warm-start checkpoint loaded")
    consistency_rules = build_consistency_rules(train_df)

    with open(output_dir / "consistency_rules.json", "w", encoding="utf-8") as file:
        json.dump(consistency_rules, file, indent=2, ensure_ascii=False)

    baseline = evaluate_loader(
        model,
        data["validation_loader"],
        device,
        tasks,
        label_maps,
        consistency_rules,
    )

    safety_thresholds = evaluation_config["safety_thresholds"]
    if not passes_safety_thresholds(
        baseline["raw_metrics"],
        safety_thresholds,
    ):
        raise RuntimeError("Warm-start safety check failed")

    best_score = model_selection_score(baseline["raw_metrics"])
    best_epoch = "v1_warm_start"

    save_checkpoint(
        output_dir / "model.pt",
        model,
        model_name,
        tasks,
        task_num_classes,
        hidden_dim,
        dropout,
        data_config["color_feature_dim"],
        label_maps,
        best_epoch,
        best_score,
        baseline["raw_metrics"],
        model_config["source_repo_id"],
    )

    criterions, weight_summary = build_criterions(
        train_df,
        tasks,
        label_maps,
        set(training_config["balanced_tasks"]),
        training_config["class_weight_min"],
        training_config["class_weight_max"],
        device,
    )

    logger.info("Class weights ready | %s", json.dumps(weight_summary))
    history = []
    started_at = time.time()

    for stage_name in ("stage1", "stage2"):
        stage = training_config[stage_name]

        active_tasks, optimizer, scheduler, scaler = (
            create_optimizer_and_scheduler(
                model,
                stage_name,
                stage,
                training_config["weight_decay"],
                len(data["train_loader"]),
                device,
                training_config["use_amp"],
            )
        )

        bad_epochs = 0
        
        for epoch in range(1, stage["epochs"] + 1):
            epoch_name = f"{stage_name}_epoch_{epoch}"

            train_loss, train_accuracy = train_one_epoch(
                model,
                data["train_loader"],
                optimizer,
                scheduler,
                scaler,
                criterions,
                active_tasks,
                device,
                training_config["use_amp"],
                training_config["max_grad_norm"],
            )

            validation = evaluate_loader(
                model,
                data["validation_loader"],
                device,
                tasks,
                label_maps,
                consistency_rules,
            )

            validation_score = model_selection_score(validation["raw_metrics"])
            safety_passed = passes_safety_thresholds(
                validation["raw_metrics"],
                safety_thresholds,
            )

            history.append(
                {
                    "epoch": epoch_name,
                    "train_loss": float(train_loss),
                    "train_accuracy": train_accuracy,
                    "validation_score": float(validation_score),
                    "safety_passed": bool(safety_passed),
                    "validation_metrics": {
                        "raw": validation["raw_metrics"],
                        "corrected": validation["corrected_metrics"],
                    },
                }
            )

            overall = validation["corrected_metrics"]["overall_metrics"]
            logger.info(
                "%s | loss=%.4f | score=%.4f | accuracy=%.4f | "
                "exact_match=%.4f | safe=%s",
                epoch_name,
                train_loss,
                validation_score,
                overall["average_accuracy"],
                overall["exact_match_accuracy"],
                safety_passed,
            )

            if safety_passed and validation_score > best_score:
                best_score = validation_score
                best_epoch = epoch_name
                bad_epochs = 0

                save_checkpoint(
                    output_dir / "model.pt",
                    model,
                    model_name,
                    tasks,
                    task_num_classes,
                    hidden_dim,
                    dropout,
                    data_config["color_feature_dim"],
                    label_maps,
                    epoch_name,
                    validation_score,
                    {
                        "raw": validation["raw_metrics"],
                        "corrected": validation["corrected_metrics"],
                    },
                    model_config["source_repo_id"],
                )

                logger.info(
                    "Improved checkpoint saved | epoch=%s",
                    epoch_name,
                )
            else:
                bad_epochs += 1

            if bad_epochs >= training_config["early_stopping_patience"]:
                logger.info("Early stopping | stage=%s", stage_name)
                break

    with open(
        output_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    best_checkpoint = torch_load(
        output_dir / "model.pt",
        map_location=device,
    )

    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.eval()

    test_results = evaluate_loader(
        model,
        data["test_loader"],
        device,
        tasks,
        label_maps,
        consistency_rules,
    )

    latency = {
        "device": device,
        "single_image": benchmark_single_image_latency(
            model,
            data["test_dataset"],
            device,
        ),
        "batch": benchmark_batch_latency(
            model,
            data["test_loader"],
            device,
        ),
    }

    final_metrics = {
        "raw": test_results["raw_metrics"],
        "corrected": test_results["corrected_metrics"],
        "latency": latency,
        "best_checkpoint": best_checkpoint["best_epoch"],
        "best_validation_score": best_checkpoint["best_validation_score"],
        "test_samples": len(data["test_dataset"]),
    }

    with open(evaluation_dir / "final_metrics.json", "w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=2, ensure_ascii=False)

    save_evaluation_artifacts(
        evaluation_dir,
        tasks,
        label_maps,
        test_results["y_true"],
        test_results["y_pred"],
        test_results["corrected_pred"],
        test_results["y_probs"],
        test_results["indices"],
    )

    save_final_metadata(
        config,
        output_dir,
        label_maps,
        task_num_classes,
        model_name,
        hidden_dim,
        dropout,
        best_checkpoint,
        final_metrics,
    )

    overall = final_metrics["corrected"]["overall_metrics"]
    training_minutes = (time.time() - started_at) / 60

    logger.info(
        "Training complete | best=%s | minutes=%.2f | "
        "accuracy=%.4f | exact_match=%.4f",
        best_epoch,
        training_minutes,
        overall["average_accuracy"],
        overall["exact_match_accuracy"],
    )

    return final_metrics