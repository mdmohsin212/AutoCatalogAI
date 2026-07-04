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

        self.heads = nn.ModuleDict(
            {
                task: ClassificationHead(
                    embedding_dim=embedding_dim,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for task, num_classes in task_num_classes.items()
            }
        )

        self.master_to_sub = nn.Linear(
            task_num_classes["masterCategory"],
            task_num_classes["subCategory"],
            bias=False,
        )

        self.sub_to_article = nn.Linear(
            task_num_classes["subCategory"],
            task_num_classes["articleType"],
            bias=False,
        )

        self.article_to_season = nn.Linear(
            task_num_classes["articleType"],
            task_num_classes["season"],
            bias=False,
        )

        self.article_to_usage = nn.Linear(
            task_num_classes["articleType"],
            task_num_classes["usage"],
            bias=False,
        )

        self.color_branch = nn.Sequential(
            nn.LayerNorm(color_feature_dim),
            nn.Linear(color_feature_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                64,
                task_num_classes["baseColour"],
            ),
        )

    def forward(self, pixel_values, color_features):
        output = self.clip.get_image_features(pixel_values=pixel_values)
        
        if hasattr(output, "pooler_output"):
            image_features = output.pooler_output
        elif isinstance(output, torch.Tensor):
            image_features = output
        else:
            image_features = output[0]
        
        image_features = F.normalize(image_features, dim=-1)
        outputs = {
            task: head(image_features)
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