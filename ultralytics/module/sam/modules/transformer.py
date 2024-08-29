import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Type

class EnhancedAttention(nn.Module):
    """
    A modified attention mechanism with optional downsampling and multi-head support.

    Attributes:
        embedding_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        downsample_rate (int): Optional downsampling factor.
    """
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = embedding_dim // downsample_rate

        assert self.internal_dim % num_heads == 0, "Number of heads must divide internal dimension evenly."

        self.q_linear = nn.Linear(embedding_dim, self.internal_dim)
        self.k_linear = nn.Linear(embedding_dim, self.internal_dim)
        self.v_linear = nn.Linear(embedding_dim, self.internal_dim)
        self.out_linear = nn.Linear(self.internal_dim, embedding_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Split tensor into multiple attention heads."""
        batch_size, seq_len, dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """Combine multi-head outputs into a single tensor."""
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """Forward pass of the attention mechanism."""
        q = self.q_linear(queries)
        k = self.k_linear(keys)
        v = self.v_linear(values)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.internal_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)

        output = self._combine_heads(output)
        return self.out_linear(output)


class DualAttentionLayer(nn.Module):
    """
    A layer performing self-attention and cross-attention in a two-directional fashion.

    Attributes:
        embedding_dim (int): Dimensionality of input embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension for MLP layers.
        activation (Type[nn.Module]): Activation function for MLP layers.
        downsample_rate (int): Downsampling rate for internal attention dimensions.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        downsample_rate: int = 2
    ) -> None:
        super().__init__()
        self.self_attention = EnhancedAttention(embedding_dim, num_heads)
        self.cross_attention = EnhancedAttention(embedding_dim, num_heads, downsample_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        """Process queries and keys with self-attention, cross-attention, and MLP."""
        # Self-attention
        q = queries + query_pe
        queries = self.self_attention(q, q, queries)
        queries = self.layer_norm1(queries)

        # Cross-attention (queries to keys)
        q = queries + query_pe
        k = keys + key_pe
        queries = self.cross_attention(q, k, keys)
        queries = self.layer_norm2(queries)

        # MLP
        mlp_out = self.mlp(queries)
        queries += mlp_out
        queries = self.layer_norm3(queries)

        # Cross-attention (keys to queries)
        k = keys + key_pe
        queries = self.cross_attention(k, queries, queries)
        keys += queries
        return queries, keys


class TwoWayTransformer(nn.Module):
    """
    A Transformer model that processes both image and query point embeddings using dual attention layers.

    Attributes:
        depth (int): Number of transformer layers.
        embedding_dim (int): Dimension of embedding vectors.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Dimension of MLP hidden layers.
        layers (nn.ModuleList): List of DualAttentionLayer instances.
        final_attention (EnhancedAttention): Final attention layer to merge queries with image embeddings.
        norm_final (nn.LayerNorm): Normalization layer applied to the final queries.
    """
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        downsample_rate: int = 2
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DualAttentionLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=mlp_dim,
                activation=activation,
                downsample_rate=downsample_rate
            ) for _ in range(depth)
        ])
        self.final_attention = EnhancedAttention(embedding_dim, num_heads, downsample_rate)
        self.norm_final = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        point_embeddings: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Process image and point embeddings through transformer layers."""
        # Flatten and permute image embeddings
        bs, c, h, w = image_embeddings.size()
        image_embeddings = image_embeddings.flatten(2).transpose(1, 2)
        image_pe = image_pe.flatten(2).transpose(1, 2)

        queries = point_embeddings
        keys = image_embeddings

        # Apply transformer layers
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embeddings, image_pe)

        # Final attention from queries to image
        q = queries + point_embeddings
        k = keys + image_pe
        attn_out = self.final_attention(q, k, keys)
        queries += attn_out
        queries = self.norm_final(queries)

        return queries, keys

"""
Class EnhancedAttention:
    Initialize:
        - embedding_dim: Dimensionality of input embeddings
        - num_heads: Number of attention heads
        - downsample_rate: Factor for downsampling dimensions

    Method _split_heads(x):
        - Reshape x to separate attention heads
        - Return reshaped tensor

    Method _combine_heads(x):
        - Combine separate heads into a single tensor
        - Return combined tensor

    Method forward(q, k, v):
        - Project q, k, v to internal dimensions
        - Split q, k, v into multiple heads
        - Compute attention scores
        - Apply softmax to attention scores
        - Multiply attention probabilities by values
        - Recombine heads
        - Return final output

Class DualAttentionLayer:
    Initialize:
        - embedding_dim: Dimensionality of embeddings
        - num_heads: Number of attention heads
        - hidden_dim: Hidden dimension for MLP
        - activation: Activation function for MLP
        - downsample_rate: Downsampling factor

    Method forward(queries, keys, query_pe, key_pe):
        - Apply self-attention to queries
        - Normalize queries
        - Apply cross-attention from queries to keys
        - Normalize queries
        - Pass queries through MLP
        - Normalize queries
        - Apply cross-attention from keys to queries
        - Update keys with attention results
        - Return updated queries and keys

Class TwoWayTransformer:
    Initialize:
        - depth: Number of transformer layers
        - embedding_dim: Dimensionality of embeddings
        - num_heads: Number of attention heads
        - mlp_dim: Hidden dimension for MLP layers
        - activation: Activation function for MLP
        - downsample_rate: Downsampling factor
        - Create list of DualAttentionLayer instances

    Method forward(image_embeddings, image_pe, point_embeddings):
        - Flatten and permute image_embeddings
        - Set queries to point_embeddings
        - Set keys to image_embeddings

        - For each layer in the list:
            - Apply layer to queries and keys

        - Apply final attention from queries to image embeddings
        - Normalize final queries
        - Return processed queries and keys

"""
