import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, heads, embed_dim):
        super(Attention, self).__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads

        assert self.head_dim * heads == embed_dim, "Embed dim must be divisible by heads"

        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.WQ(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        K = self.WK(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        V = self.WV(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(heads, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, layers, heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, mlp_dim, heads, dropout) for _ in range(layers)
        ])

    def forward(self, x):
        return self.blocks(x)

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, mlp_dim, num_layers,
                 num_classes, num_heads, dropout=0.1, in_channels=3):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.linear_proj = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(embed_dim, mlp_dim, num_layers, num_heads, dropout)
        self.mlp_head = ClassificationHead(embed_dim, num_classes, dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, self.num_patches, -1)
        x = self.linear_proj(patches)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        return self.mlp_head(x[:, 0]), x