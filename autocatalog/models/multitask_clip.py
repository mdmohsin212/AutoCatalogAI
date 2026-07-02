import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from autocatalog.models.heads import ClassificationHead

class CLIPMultiTaskClassifier(nn.Module):
    def __init__(self, model_name, task_num_classes, hidden_dim=512, droput=0.2, unfreeze_last_n_vision_layers=0):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        
        for param in self.clip.parameters():
            param.requires_grad = False
        
        if unfreeze_last_n_vision_layers > 0:
            vision_layer = self.clip.vision_model.encoder.layers
            
            for layer in vision_layer[-unfreeze_last_n_vision_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
                
            for param in self.clip.visual_projection.parameters():
                param.requires_grad = True
            
            for param in self.clip.vision_model.post_layernorm.parameters():
                param.requires_grad = True
        
        embedding_dim = self.clip.config.projection_dim
        self.heads = nn.ModuleDict({
            task : ClassificationHead(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=droput,
            )
            for task, num_classes in task_num_classes.items()
        })

    
    def forward(self, pixel_values):
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)

        return {
            task : head(image_features)
            for task, head in self.heads.items()
        }