from functools import partial
import torch

from ultralytics.utils.downloads import attempt_download_asset
from .modules.decoders import MaskDecoder
from .modules.encoders import ImageEncoderViT, PromptEncoder
from .modules.sam import Sam
from .modules.tiny_encoder import TinyViT
from .modules.transformer import TwoWayTransformer


def create_sam_vit_h_model(checkpoint=None):
    """Create and return a high-resolution Segment Anything Model (SAM)."""
    return initialize_sam_model(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        global_attn_indexes=[7, 15, 23, 31],
        checkpoint_path=checkpoint,
    )


def create_sam_vit_l_model(checkpoint=None):
    """Create and return a large Segment Anything Model (SAM)."""
    return initialize_sam_model(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        global_attn_indexes=[5, 11, 17, 23],
        checkpoint_path=checkpoint,
    )


def create_sam_vit_b_model(checkpoint=None):
    """Create and return a base Segment Anything Model (SAM)."""
    return initialize_sam_model(
        embed_dim=768,
        depth=12,
        num_heads=12,
        global_attn_indexes=[2, 5, 8, 11],
        checkpoint_path=checkpoint,
    )


def create_mobile_sam_model(checkpoint=None):
    """Create and return a Mobile Segment Anything Model (Mobile-SAM)."""
    return initialize_sam_model(
        embed_dim=[64, 128, 160, 320],
        depth=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        global_attn_indexes=None,
        is_mobile_sam=True,
        checkpoint_path=checkpoint,
    )


def initialize_sam_model(
    embed_dim, depth, num_heads, global_attn_indexes, checkpoint_path=None, is_mobile_sam=False
):
    """Initialize the SAM model based on specified configurations."""
    prompt_embedding_dim = 256
    image_dim = 1024
    vit_patch_size = 16
    embedding_size = image_dim // vit_patch_size

    # Choose between TinyViT or ImageEncoderViT based on mobile_sam flag
    encoder = (
        TinyViT(
            img_size=image_dim,
            in_chans=3,
            num_classes=1000,
            embed_dims=embed_dim,
            depths=depth,
            num_heads=num_heads,
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )
        if is_mobile_sam
        else ImageEncoderViT(
            depth=depth,
            embed_dim=embed_dim,
            img_size=image_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=global_attn_indexes,
            window_size=14,
            out_chans=prompt_embedding_dim,
        )
    )

    # Create SAM model components
    sam_model = Sam(
        image_encoder=encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embedding_dim,
            image_embedding_size=(embedding_size, embedding_size),
            input_image_size=(image_dim, image_dim),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embedding_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embedding_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_file = attempt_download_asset(checkpoint_path)
        with open(checkpoint_file, "rb") as file:
            state_dict = torch.load(file)
        sam_model.load_state_dict(state_dict)

    sam_model.eval()  # Set model to evaluation mode
    return sam_model


# Mapping of checkpoint files to their corresponding model creation functions
model_factory_map = {
    "sam_h.pt": create_sam_vit_h_model,
    "sam_l.pt": create_sam_vit_l_model,
    "sam_b.pt": create_sam_vit_b_model,
    "mobile_sam.pt": create_mobile_sam_model,
}


def build_sam_model(checkpoint_name="sam_b.pt"):
    """Build the SAM model specified by the checkpoint file name."""
    model_builder = None
    checkpoint_name = str(checkpoint_name)  # Convert checkpoint name to string if it's a Path object
    
    # Find the appropriate model creation function based on checkpoint file name
    for key in model_factory_map.keys():
        if checkpoint_name.endswith(key):
            model_builder = model_factory_map[key]
            break

    if not model_builder:
        raise FileNotFoundError(f"Unsupported checkpoint file: {checkpoint_name}. Available models: {model_factory_map.keys()}")

    return model_builder(checkpoint_name)


"""
# Pseudocode for Building and Loading SAM Models

Define function create_sam_vit_h_model(checkpoint=None):
    # Create and return a high-resolution SAM model
    Return initialize_sam_model(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        global_attn_indexes=[7, 15, 23, 31],
        checkpoint_path=checkpoint
    )

Define function create_sam_vit_l_model(checkpoint=None):
    # Create and return a large SAM model
    Return initialize_sam_model(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        global_attn_indexes=[5, 11, 17, 23],
        checkpoint_path=checkpoint
    )

Define function create_sam_vit_b_model(checkpoint=None):
    # Create and return a base SAM model
    Return initialize_sam_model(
        embed_dim=768,
        depth=12,
        num_heads=12,
        global_attn_indexes=[2, 5, 8, 11],
        checkpoint_path=checkpoint
    )

Define function create_mobile_sam_model(checkpoint=None):
    # Create and return a Mobile SAM model
    Return initialize_sam_model(
        embed_dim=[64, 128, 160, 320],
        depth=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        global_attn_indexes=None,
        is_mobile_sam=True,
        checkpoint_path=checkpoint
    )

Define function initialize_sam_model(embed_dim, depth, num_heads, global_attn_indexes, checkpoint_path=None, is_mobile_sam=False):
    # Initialize the SAM model based on provided parameters

    # Set fixed parameters
    prompt_embedding_dim = 256
    image_dim = 1024
    vit_patch_size = 16
    embedding_size = image_dim // vit_patch_size

    # Select the appropriate image encoder based on is_mobile_sam flag
    If is_mobile_sam:
        Initialize encoder as TinyViT with specified parameters
    Else:
        Initialize encoder as ImageEncoderViT with specified parameters

    # Create the SAM model with the encoder, prompt encoder, and mask decoder
    sam_model = Create Sam model with:
        - image_encoder
        - prompt_encoder with prompt_embedding_dim and image_embedding_size
        - mask_decoder with MaskDecoder and TwoWayTransformer
        - pixel_mean and pixel_std for normalization

    # Load checkpoint if provided
    If checkpoint_path is not None:
        Download checkpoint file
        Load model state from checkpoint file
        Set model to evaluation mode

    Return sam_model

# Mapping of checkpoint filenames to model creation functions
Define model_factory_map as:
    "sam_h.pt" -> create_sam_vit_h_model
    "sam_l.pt" -> create_sam_vit_l_model
    "sam_b.pt" -> create_sam_vit_b_model
    "mobile_sam.pt" -> create_mobile_sam_model

Define function build_sam_model(checkpoint_name="sam_b.pt"):
    # Build the SAM model specified by the checkpoint file name

    # Convert checkpoint_name to string if necessary
    For each key in model_factory_map:
        If checkpoint_name ends with key:
            Set model_builder to model_factory_map[key]
            Break loop

    If model_builder is not found:
        Raise error with message about unsupported checkpoint file

    Return model_builder with checkpoint_name
"""
