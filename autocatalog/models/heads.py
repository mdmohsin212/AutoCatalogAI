import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)