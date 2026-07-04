import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from autocatalog.models.heads import ClassificationHead


class CLIPMultiTaskClassifierV2(nn.Module):
    def __init__(self, model_name, task_num_classes, hidden_dim=512, dropout=0.2, color_feature_dim=37):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
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
        output = self.clip.get_image_features(pixel_values=pixel_values)
        
        if hasattr(output, "pooler_output"):
            image_features = output.pooler_output
        elif isinstance(output, torch.Tensor):
            image_features = output
        else:
            image_features = output[0]
        
        image_features = F.normalize(image_features, dim=-1)
        
        return {
            task : head(image_features)
            for task, head in self.heads.items()
        }

        master_probs = torch.softmax(outputs["masterCategory"].detach(),dim=1)
        
        outputs["subCategory"] = (outputs["subCategory"] + self.master_to_sub(master_probs))
        sub_probs = torch.softmax(outputs["subCategory"].detach(), dim=1)
        
        outputs["articleType"] = (outputs["articleType"] + self.sub_to_article(sub_probs))
        article_probs = torch.softmax(outputs["articleType"].detach(), dim=1)

        outputs["season"] = (outputs["season"] + self.article_to_season(article_probs))

        outputs["usage"] = (outputs["usage"] + self.article_to_usage(article_probs))

        outputs["baseColour"] = (outputs["baseColour"] + self.color_branch(color_features))

        return outputs