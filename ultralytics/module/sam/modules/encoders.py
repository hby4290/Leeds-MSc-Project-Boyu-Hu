def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of query and key sizes.

    Args:
        q_size (int): Size of query q.
        k_size (int): Size of key k.
        rel_pos (Tensor): Relative position embeddings (L, C).

    Returns:
        torch.Tensor: Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Assuming 'rel_pos' shape is (L, C) where L is the max distance
        # Create a grid for relative positions
        rel_pos_grid = torch.arange(-q_size + 1, k_size).unsqueeze(1)
        rel_pos_grid = rel_pos_grid.expand(-1, k_size)
        # Interpolate
        rel_pos = F.interpolate(rel_pos.unsqueeze(0), size=(rel_pos_grid.shape[0],), mode='linear', align_corners=False)
        rel_pos = rel_pos.squeeze(0)

    # Extract relative positional embeddings
    pos = rel_pos[q_size - 1 : q_size + k_size - 1, :]
    return pos

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_hw: Tuple[int, int],
    k_hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Add decomposed relative positional embeddings to attention scores.

    Args:
        attn (torch.Tensor): Attention scores with shape (B * nHead, H * W, H * W).
        q (torch.Tensor): Query tensor with shape (B * nHead, H * W, C).
        rel_pos_h (torch.Tensor): Relative position embeddings for height.
        rel_pos_w (torch.Tensor): Relative position embeddings for width.
        q_hw (Tuple[int, int]): Height and width of the query.
        k_hw (Tuple[int, int]): Height and width of the key.

    Returns:
        torch.Tensor: Attention scores with added relative positional embeddings.
    """
    B, H, W, C = q.shape
    # Create relative position grids
    rel_pos_h = get_rel_pos(q_hw[0], k_hw[0], rel_pos_h)
    rel_pos_w = get_rel_pos(q_hw[1], k_hw[1], rel_pos_w)
    rel_pos = rel_pos_h.unsqueeze(0).unsqueeze(1) + rel_pos_w.unsqueeze(0).unsqueeze(1)
    
    # Add relative positional embeddings to attention scores
    attn = attn + (q @ rel_pos).view(B, H, W, C).sum(dim=-1)
    return attn

"""
Class ImageEncoderViT:
    Initialize(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans, qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size, global_attn_indexes):
        Set input image size to img_size
        Create PatchEmbed module to split the image into patches and embed them
        If use_abs_pos:
            Initialize absolute position encoding
        
        For each layer in range(depth):
            Create Transformer Block and add to blocks list
        
        Create neck module including two convolution layers and normalization layer

    Forward(x):
        Pass input image x through PatchEmbed
        If use_abs_pos:
            Add absolute position encoding to x
        
        For each block in blocks:
            Pass x through the block
        
        Pass x through neck module
        Return the output

"""
