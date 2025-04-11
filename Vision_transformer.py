import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class Attention(nn.Module):
    '''
    Attention Module is used to perform self-attention operation allowing the
    model to attend information from different representation subspaces on an input
    sequence of embeddings.

    Args:
        embed_dim: Dimension size of the hidden embedding
        heads: Number of parallel attention heads
    Methods:
        forward(inp) :-
        Performs the self-attention operation on the input sequence embedding.
        Returns the output of self-attention can be seen as an attention map
        inp (batch_size, seq_len, embed_dim)
        out:(batch_size, seq_len, embed_dim)
    Examples:
    >>> attention = Attention(embed_dim, heads, activation, dropout)
    >>> out = attention(inp)
    '''

    def __init__(self, heads, embed_dim):
        self.heads = heads
        self.embed_dim = embed_dim
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(embed_dim, embed_dim)) for _ in range(3)])

        super(Attention, self).__init__()

    def forward(self, inp):
        '''
        Args:
            inp: Input sequence embedding of shape (batch_size, seq_len, embed_dim)
        Returns:
            out: Output of self-attention of shape (batch_size, seq_len, embed_dim)
        '''
        # inp: (batch_size, seq_len, embed_dim)
        # out: (batch_size, seq_len, embed_dim)
        # Step 1: Compute Q = XWQ, K=XWK, V = XWV where WQ, WK, WV are trainable weights of size (embed_dim, embed_dim)
        Q, K, V = [inp @ w for w in self.weights]
        # Step 2: Resize Q, K, V to size (batch_size, seq_len, heads,embed_dim // heads) where heads are number of heads, and then permute it to size (batch_size, heads, seq_len, embed_dim // heads)
        Q = Q.view(Q.shape[0], Q.shape[1], self.heads, self.embed_dim // self.heads).permute(0, 2, 1, 3)
        K = K.view(K.shape[0], K.shape[1], self.heads, self.embed_dim // self.heads).permute(0, 2, 1, 3)
        V = V.view(V.shape[0], V.shape[1], self.heads, self.embed_dim // self.heads).permute(0, 2, 1, 3)
        #  Step 3: Compute attention 𝑎𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑄𝐾𝑇√(𝑒𝑚𝑏𝑒𝑑_ dim // ℎ𝑒𝑎𝑑𝑠))*V
        attention = torch.softmax(Q @ K.transpose(-2, -1) / (self.embed_dim // self.heads) ** 0.5, dim=-1) 
        attention = attention @ V  # attention is of size (batch_size, heads, seq_len, embed_dim // heads)
        # Step 4: return output = attention . V, Output is of size (batch_size, seq_len, embed_dim)
        return attention.permute(0, 2, 1, 3).contiguous().view(attention.shape[0], attention.shape[1], self.embed_dim)
    

class TransformerBlock(nn.Module):
    '''
    Transformer Block combines both the attention module and the feed-forward
    module with layer normalization, dropout and residual connections. The sequence
    of operations is as follows :-

    Inp -> LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FeedForward -> Out
    | | | |
    |-------------Addition--------------| |---------------Addition------------|
    Args:
        embed_dim: Dimension size of the hidden embedding
        heads: Number of parallel attention heads (Default=8)
    mlp_dim: The higher dimension is used to transform the input embedding
    and then resized back to embedding dimension to capture richer information.
    dropout: Dropout value for the layer on attention_scores (Default=0.1)

    Methods:
    forward(inp) :-
    Applies the sequence of operations mentioned above.
    (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
    Examples:
    >> TB = TransformerBlock(embed_dim, mlp_dim, heads, activation, dropout)
    >> out = TB(inp)
    '''
    def __init__(self, embed_dim, mlp_dim, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(heads, embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        # out: (batch_size, seq_len, embed_dim)
        # Start inp -> LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FeedForward -> Out
        # Step 1: Apply LayerNorm1 to the input sequence embedding
        norm1 = self.norm1(inp)  # norm1: (batch_size, seq_len, embed_dim)
        # Step 2: Apply self-attention to the normalized input sequence embedding
        attention = self.attention(norm1, norm1, norm1)[0]  # attention: (batch_size, seq_len, embed_dim)
        # Step 3: Apply residual connection to the attention output and the input sequence embedding
        attention = inp + self.dropout1(attention)  # attention: (batch_size, seq_len, embed_dim)
        # Step 4: Apply LayerNorm2 to the attention output
        norm2 = self.norm2(attention)  # norm2: (batch_size, seq_len, embed_dim)
        # Step 5: Apply FeedForward to the normalized attention output
        ff = self.fc2(self.activation(self.fc1(norm2)))  # ff: (batch_size, seq_len, embed_dim)
        # Step 6: Apply residual connection to the feedforward output and the attention output
        ff = attention + self.dropout2(ff)  # ff: (batch_size, seq_len, embed_dim)
        # Step 7: Return the output of the transformer block
        return ff  # ff: (batch_size, seq_len, embed_dim)


class Transformer(nn.Module):
    '''
    Transformer combines multiple layers of Transformer Blocks in a sequential
    manner. The sequence
    of the operations is as follows -
    Input -> TB1 -> TB2 -> .......... -> TBn (n being the number of layers) ->
    Output
    Args:
    embed_dim: Dimension size of the hidden embedding in the TransfomerBlock
    mlp_dim: Dimension size of MLP layer in the TransfomerBlock
    layers: Number of Transformer Blocks in the Transformer
    heads: Number of parallel attention heads (Default=8)
    dropout: Dropout value for the layer on attention_scores (Default=0.1)

    Methods:
    forward(inp) :-
    Applies the sequence of operations mentioned above.
    (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
    Examples:
        transformer = Transformer(embed_dim, layers, heads, activation,
        forward_expansion, dropout)
        out = transformer(inp)
    '''
    def __init__(self, embed_dim, layers, heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.trans_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, mlp_dim, heads, dropout)
            for i in range(layers)]
        )

    def forward(self, inp):
        pass    
    # inp: (batch_size, seq_len, embed_dim)


class ClassificationHead(nn.Module):
    '''
    Classification Head attached to the first sequence token which is used as
    the arbitrary
    classification token and used to optimize the transformer model by applying
    Cross-Entropy
    loss. The sequence of operations is as follows :-
    Input -> FC1 -> GELU -> Dropout -> FC2 -> Output
    Args:
    embed_dim: Dimension size of the hidden embedding
    classes: Number of classification classes in the dataset
    dropout: Dropout value for the layer on attention_scores (Default=0.1)
    Methods:
    forward(inp) :-
    Applies the sequence of operations mentioned above.
    (batch_size, embed_dim) -> (batch_size, classes)
    Examples:
    CH = ClassificationHead(embed_dim, classes, dropout)
    out = CH(inp)
    '''
    def __init__(self, embed_dim, classes, dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.classes = classes
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim // 2, classes)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        pass
        # inp: (batch_size, emb)


class VisionTransformer(nn.Module):
    '''
    Vision Transformer is the complete end-to-end model architecture that
    combines all the above modules in a sequential manner. The sequence of the
    operations is as follows -

    Args:
    patch_size: Length of square patch size
    max_len: Max length of learnable positional embedding
    embed_dim: Dimension size of the hidden embedding
    mlp_dim: Dimension size of MLP embedding
    classes: Number of classes in the dataset
    layers: Number of Transformer Blocks in the Transformer
    channels: Number of channels in the input (Default=3)
    heads: Number of parallel attention heads (Default=8)

    dropout: Dropout value for the layer on attention_scores (Default=0.1)

    Methods:
    forward(inp) :-
    Applies the sequence of operations mentioned above.
    It outputs the classification output as well as the sequence output of
    the transformer
    (batch_size, channels, width, height) -> (batch_size, classes),
    (batch_size, seq_len+1, embed_dim)
    
    Examples:
        ViT = VisionTransformer(inp_channels, patch_size, max_len, heads,
    classes, layers, embed_dim, mlp_dim, channels,dropout)
      class_out, hidden_seq = ViT(inp)
    '''
    def __init__(self, inp_channels, patch_size, max_len, heads, classes,layers, embed_dim, mlp_dim, dropout):
        super(VisionTransformer, self).__init__()
    
    def forward(self, inp):
        pass
        # inp: (batch_size, channels, width, height)
