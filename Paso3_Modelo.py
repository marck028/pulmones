import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PatchEmbedCNN(nn.Module):
    """CNN como reductor de tokens para mitigar el costo cuadrático O(N^2)"""
    def __init__(self, img_size=224, embed_dim=256):
        super().__init__()
        base_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Extraemos el backbone convolucional sin pooling ni fully connected
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-2]) 
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.num_patches = (img_size // 32) * (img_size // 32) # 7x7 = 49 patches

    def forward(self, x):
        x = self.backbone(x)            
        x = self.proj(x)                
        x = x.flatten(2).transpose(1, 2) # Formato: [Batch, Tokens, Embed_Dim]
        return x

class HybridCNNViT(nn.Module):
    """Modelo híbrido CNN-ViT con mecanismos de autoatención"""
    def __init__(self, img_size=224, num_classes=4, embed_dim=256, depth=4, heads=8, mlp_dim=512, dropout=0.2):
        super().__init__()
        self.patch_embed = PatchEmbedCNN(img_size=img_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Token CLS y Embeddings de posición geométrica
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Capas de Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=mlp_dim, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) 
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0]) # Clasificación final basada en el CLS token