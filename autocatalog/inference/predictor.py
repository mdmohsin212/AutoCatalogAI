import json
import time
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPImageProcessor

from autocatalog.models.multitask_clip import CLIPMultiTaskClassifier
from autocatalog.inference.catalog_generator import generate_catalog_output


class AutoCatalogPredictor:
    def __init__(
        self,
        repo_id="mohsin416/autocatalogai-clip-multitask",
        device=None,
        top_k=3,
    ):
        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        
        self.model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="model.pt",
            repo_type="model"
        )
        
        self.config_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="config.json",
            repo_type="model"
        )
        
        self.label_map = hf_hub_download(
            repo_id=self.repo_id,
            filename="label_maps.json",
            repo_type="model"
        )
        
        self.metrics_path = self._try_download("metrics.json")
        
        self.config = self._load_json(self.config_path)
        self.label_maps = self._load_json(self.label_maps_path)
        self.metrics = self._load_json(self.metrics_path) if self.metrics_path else {}
        
        self.tasks = self.config.get("tasks") or list(self.label_maps.keys())
        self.base_model_name = self.config.get("base_model_name") or self.config.get("model_name")
        self.hidden_dim = self.config.get("hidden_dim", 512)
        self.dropout = self.config.get("dropout", 0.2)
        self.unfreeze_last_n_vision_layers = self.config.get("unfreeze_last_n_vision_layers", 0)
        
        if self.base_model_name is None:
            self.base_model_name = "openai/clip-vit-base-patch32"
        
        self.task_num_classes = self._get_task_num_classes()
        self.processor = CLIPImageProcessor.from_pretrained(self.base_model_name)
        self.model = self._load_model()
        self.model.eval()
    
    
    def _try_download(self, filename):
        try:
            return hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="model",
            )
        except Exception:
            return None
    
    
    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    
    def _safe_torch_load(self, path):
        try:
            return torch.load(
                path,
                map_location=self.device,
                weights_only=False
            )
        
        except TypeError:
            return torch.load(
                path,
                map_location=self.device
            )
    
    def _get_task_num_classes(self):
        if "task_num_classes" in self.config:
            return {
                task: int(value)
                for task, value in self.config["task_num_classes"].items()
            }
        
        task_num_classes = {}
        for task in self.tasks:
            task_num_classes[task] = len(self.label_maps[task]["label2id"])
            
        return task_num_classes
    
    def _load_model(self):
        checkpoint = self._safe_torch_load(self.model_path)
        
        model = CLIPMultiTaskClassifier(
            model_name=self.base_model_name,
            task_num_classes=self.task_num_classes,
            hidden_dim=self.hidden_dim,
            droput=self.dropout,
            unfreeze_last_n_vision_layers=self.unfreeze_last_n_vision_layers
        )
        
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        
    
    def _prepare_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        
    
    def predict(self, image, top_k=None):
        top_k = top_k or self.top_k
        image = self._prepare_image(image)
        
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        prediction = {}
        simple_predictions = {}
        
        for task in self.tasks:
            logits = outputs[task]
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            
            k = min(top_k, probs.shape[0])
            top_probs, top_indices = torch.topk(
                probs,
                k=k
            )
            
            top_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.label_maps[task]["id2label"][str(int(idx.item()))] 
                
                top_predictions.append({
                    "label": label,
                    "confidence": float(prob.item()),
                })
            
            prediction[task] = {
                "label": top_predictions[0]["label"],
                "confidence": top_predictions[0]["confidence"],
                "top_3": top_predictions,
            }

            simple_predictions[task] = top_predictions[0]["label"]

        catalog_output = generate_catalog_output(simple_predictions)

        return {
            "prediction": prediction,
            "catalog_output": catalog_output,
            "runtime": {
                "device": self.device,
                "inference_time_ms": float((end_time - start_time) * 1000),
                "model": self.base_model_name,
                "repo_id": self.repo_id,
            },
        }

    def get_model_metrics(self):
        return self.metrics