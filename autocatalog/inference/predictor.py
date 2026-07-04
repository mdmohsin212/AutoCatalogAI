import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPImageProcessor

from autocatalog.data.preprocessing import extract_color_features
from autocatalog.inference.catalog_generator import generate_catalog_output
from autocatalog.models.multitask_clip import CLIPMultiTaskClassifierV2
from autocatalog.utils.logger import get_logger


logger = get_logger(__name__)


class AutoCatalogPredictor:
    def __init__(
        self,
        repo_id="mohsin416/autocatalogai-clip-multitask-v2",
        device=None,
        top_k=3,
        apply_consistency_rules=True,
    ):
        self.repo_id = repo_id
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.top_k = top_k
        self.apply_consistency_rules = apply_consistency_rules

        logger.info(
            "Loading V2 model from Hugging Face | repo=%s | device=%s",
            self.repo_id,
            self.device,
        )

        self.model_path = self._download("model.pt")
        self.config = self._load_json(
            self._download("config.json")
        )
        self.label_maps = self._load_json(
            self._download("label_maps.json")
        )
        self.consistency_rules = self._load_json(
            self._download("consistency_rules.json")
        )
        self.metrics = self._load_json(
            self._download("metrics.json")
        )

        self.tasks = self.config["tasks"]
        self.model_name = self.config["base_model_name"]

        self.processor = CLIPImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = self._load_model()

        logger.info(
            "V2 model loaded successfully | tasks=%d",
            len(self.tasks),
        )

    def _download(self, filename):
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
        )

    @staticmethod
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_checkpoint(path):
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(
                path,
                map_location="cpu",
            )

    def _load_model(self):
        checkpoint = self._load_checkpoint(
            self.model_path
        )

        model = CLIPMultiTaskClassifierV2(
            model_name=checkpoint["model_name"],
            task_num_classes=checkpoint["task_num_classes"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
            color_feature_dim=checkpoint["color_feature_dim"],
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        model.to(self.device)
        model.eval()

        return model

    @staticmethod
    def _prepare_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")

        return Image.open(image).convert("RGB")

    def _apply_rules(self, predicted_ids, probabilities):
        corrected_ids = predicted_ids.copy()
        corrections = []

        article_id = predicted_ids["articleType"]

        article_label = self.label_maps[
            "articleType"
        ]["id2label"][str(article_id)]

        article_confidence = probabilities[
            "articleType"
        ][article_id]

        if article_confidence < 0.65:
            return corrected_ids, corrections

        mappings = [
            (
                "article_to_master",
                "masterCategory",
                0.95,
            ),
            (
                "article_to_sub",
                "subCategory",
                0.90,
            ),
            (
                "article_to_usage",
                "usage",
                0.92,
            ),
            (
                "article_to_season",
                "season",
                0.92,
            ),
        ]

        for rule_name, target_task, minimum_dominance in mappings:
            rule = self.consistency_rules[
                rule_name
            ].get(article_label)

            if not rule:
                continue

            if rule["dominance"] < minimum_dominance:
                continue

            target_label = rule["target"]

            target_id = self.label_maps[
                target_task
            ]["label2id"][target_label]

            old_id = corrected_ids[target_task]

            if old_id == target_id:
                continue

            old_label = self.label_maps[
                target_task
            ]["id2label"][str(old_id)]

            corrected_ids[target_task] = target_id

            corrections.append(
                {
                    "task": target_task,
                    "from": old_label,
                    "to": target_label,
                }
            )

        return corrected_ids, corrections

    @torch.inference_mode()
    def predict(
        self,
        image,
        top_k=None,
        apply_consistency_rules=None,
    ):
        started_at = time.perf_counter()

        image = self._prepare_image(image)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].to(self.device)

        color_features = torch.tensor(
            extract_color_features(image),
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_started_at = time.perf_counter()

        outputs = self.model(
            pixel_values,
            color_features,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_time_ms = (
            time.perf_counter() - inference_started_at
        ) * 1000

        probabilities = {}
        predicted_ids = {}

        for task in self.tasks:
            task_probs = torch.softmax(
                outputs[task],
                dim=1,
            )[0].cpu().numpy()

            probabilities[task] = task_probs
            predicted_ids[task] = int(
                np.argmax(task_probs)
            )

        use_rules = (
            self.apply_consistency_rules
            if apply_consistency_rules is None
            else apply_consistency_rules
        )

        final_ids = predicted_ids.copy()
        corrections = []

        if use_rules:
            final_ids, corrections = self._apply_rules(
                predicted_ids,
                probabilities,
            )

        selected_top_k = top_k or self.top_k

        prediction = {}
        simple_predictions = {}

        for task in self.tasks:
            task_probs = probabilities[task]

            k = min(
                selected_top_k,
                len(task_probs),
            )

            top_indices = np.argsort(
                task_probs
            )[-k:][::-1]

            top_items = [
                {
                    "label": self.label_maps[
                        task
                    ]["id2label"][str(int(index))],
                    "confidence": float(
                        task_probs[index]
                    ),
                }
                for index in top_indices
            ]

            final_id = final_ids[task]
            raw_id = predicted_ids[task]

            final_label = self.label_maps[
                task
            ]["id2label"][str(final_id)]

            raw_label = self.label_maps[
                task
            ]["id2label"][str(raw_id)]

            prediction[task] = {
                "label": final_label,
                "confidence": float(
                    task_probs[final_id]
                ),
                "top_3": top_items,
                "corrected": final_id != raw_id,
                "raw_label": (
                    raw_label
                    if final_id != raw_id
                    else None
                ),
            }

            simple_predictions[task] = final_label

        total_time_ms = (
            time.perf_counter() - started_at
        ) * 1000

        logger.info(
            "Prediction completed | inference_ms=%.2f | total_ms=%.2f",inference_time_ms,total_time_ms,)

        return {
            "prediction": prediction,
            "corrections": corrections,
            "catalog_output": generate_catalog_output(
                simple_predictions
            ),
            "runtime": {
                "device": str(self.device),
                "inference_time_ms": float(
                    inference_time_ms
                ),
                "total_time_ms": float(
                    total_time_ms
                ),
                "model": self.model_name,
                "repo_id": self.repo_id,
            },
        }

    def get_model_metrics(self):
        return self.metrics