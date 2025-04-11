import torch
import torch.nn as nn
import math

# Attention mechanism for the Transformer
class Attention(nn.Module):
    def __init__(self, heads, embed_dim):
        """
        Initializes the Attention module.

        Args:
            heads (int): Number of attention heads.
            embed_dim (int): Dimensionality of the input embeddings.
        """
        super(Attention, self).__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads

        # Ensure the embedding dimension is divisible by the number of heads
        assert self.head_dim * heads == embed_dim, "Embed dim must be divisible by heads"

        # Linear layers for query, key, and value projections
        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, embed_dim).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, N, D = x.shape
        # Compute query, key, and value matrices
        Q = self.WQ(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        K = self.WK(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        V = self.WV(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out(out)

# Transformer block consisting of attention and MLP layers
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, heads, dropout=0.1):
        """
        Initializes the TransformerBlock module.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            mlp_dim (int): Dimensionality of the MLP hidden layer.
            heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
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
        """
        Forward pass for the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, embed_dim).

        Returns:
            torch.Tensor: Output tensor after applying the transformer block.
        """
        x = x + self.attn(self.norm1(x))  # Apply attention with residual connection
        x = x + self.mlp(self.norm2(x))  # Apply MLP with residual connection
        return x

# Transformer consisting of multiple Transformer blocks
class Transformer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, layers, heads, dropout=0.1):
        """
        Initializes the Transformer module.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            mlp_dim (int): Dimensionality of the MLP hidden layer.
            layers (int): Number of Transformer blocks.
            heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(Transformer, self).__init__()
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, mlp_dim, heads, dropout) for _ in range(layers)
        ])

    def forward(self, x):
        """
        Forward pass for the Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, embed_dim).

        Returns:
            torch.Tensor: Output tensor after applying the Transformer.
        """
        return self.blocks(x)

# Classification head for the Vision Transformer
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        """
        Initializes the ClassificationHead module.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(ClassificationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for the ClassificationHead.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.mlp(x)

# Vision Transformer (ViT) model
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, mlp_dim, num_layers,
                 num_classes, num_heads, dropout=0.1, in_channels=3):
        """
        Initializes the VisionTransformer module.

        Args:
            img_size (int): Size of the input image (assumes square images).
            patch_size (int): Size of each patch (assumes square patches).
            embed_dim (int): Dimensionality of the input embeddings.
            mlp_dim (int): Dimensionality of the MLP hidden layer.
            num_layers (int): Number of Transformer blocks.
            num_classes (int): Number of output classes.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
        """
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection of flattened patches
        self.linear_proj = nn.Linear(self.patch_dim, embed_dim)
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.transformer = Transformer(embed_dim, mlp_dim, num_layers, num_heads, dropout)
        # Classification head
        self.mlp_head = ClassificationHead(embed_dim, num_classes, dropout)

    def forward(self, x):
        """
        Forward pass for the VisionTransformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, img_size, img_size).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
            torch.Tensor: Final embeddings of shape (batch_size, embed_dim).
        """
        B, C, H, W = x.shape
        # Extract patches from the image
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, self.num_patches, -1)
        x = self.linear_proj(patches)

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Pass through the Transformer encoder
        x = self.transformer(x)
        # Classification output
        return self.mlp_head(x[:, 0]), x