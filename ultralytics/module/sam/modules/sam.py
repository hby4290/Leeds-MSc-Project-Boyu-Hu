from typing import List, Tuple

import torch
from torch import nn

from .decoders import MaskDecoder
from .encoders import ImageEncoderViT, PromptEncoder

class SegmentModel(nn.Module):
    """
    SegmentModel is designed to perform object segmentation. It processes images to generate embeddings using 
    a visual encoder, and encodes prompts using a prompt encoder. These components work together with a mask
    decoder to generate segmentation masks based on the embeddings.

    Attributes:
        mask_threshold (float): The threshold for the mask prediction, to determine which masks are considered valid.
        image_format (str): Format of the input images (e.g., 'RGB').
        encoder (ImageEncoderViT): Encoder that converts images into embeddings.
        prompt_encoder (PromptEncoder): Encodes various input prompts into embeddings.
        decoder (MaskDecoder): Converts combined embeddings into segmentation masks.
        normalization_mean (List[float]): Mean values for image normalization.
        normalization_std (List[float]): Standard deviation values for image normalization.
    """

    mask_threshold: float = 0.5  # Updated default threshold for better mask prediction
    image_format: str = "RGB"

    def __init__(
        self,
        encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder: MaskDecoder,
        normalization_mean: List[float] = [123.675, 116.28, 103.53],
        normalization_std: List[float] = [58.395, 57.12, 57.375]
    ) -> None:
        """
        Initializes the SegmentModel with the necessary components for segmentation.

        Note:
            The forward pass logic has been updated to a separate method.

        Args:
            encoder (ImageEncoderViT): Encoder to process images into embeddings.
            prompt_encoder (PromptEncoder): Encodes prompts into embeddings.
            decoder (MaskDecoder): Decodes the combined embeddings into segmentation masks.
            normalization_mean (List[float], optional): Mean pixel values for normalization. Defaults to [123.675, 116.28, 103.53].
            normalization_std (List[float], optional): Std pixel values for normalization. Defaults to [58.395, 57.12, 57.375].
        """
        super().__init__()
        self.encoder = encoder
        self.prompt_encoder = prompt_encoder
        self.decoder = decoder
        self.register_buffer("normalization_mean", torch.Tensor(normalization_mean).view(1, -1, 1, 1))
        self.register_buffer("normalization_std", torch.Tensor(normalization_std).view(1, -1, 1, 1))

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input image tensor using mean and std values.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        image = (image - self.normalization_mean) / self.normalization_std
        return image

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the model to predict object masks.

        Args:
            image (torch.Tensor): The input image tensor.
            prompt (torch.Tensor): The input prompt tensor.

        Returns:
            torch.Tensor: Predicted masks tensor.
        """
        # Preprocess the image
        normalized_image = self.preprocess_image(image)
        
        # Generate image embeddings
        image_embeddings = self.encoder(normalized_image)
        
        # Encode the prompt
        prompt_embeddings = self.prompt_encoder(prompt)
        
        # Combine embeddings and decode to get masks
        combined_embeddings = torch.cat([image_embeddings, prompt_embeddings], dim=1)
        masks = self.decoder(combined_embeddings)
        
        # Apply threshold to the masks
        masks = (masks > self.mask_threshold).float()
        
        return masks

"""
Class SegmentModel:
    Attributes:
        mask_threshold (float): Threshold for mask validity.
        image_format (str): Format of the input image.
        encoder (ImageEncoderViT): Image encoder for generating embeddings.
        prompt_encoder (PromptEncoder): Encoder for processing input prompts.
        decoder (MaskDecoder): Decoder for creating masks from embeddings.
        normalization_mean (List[float]): Mean values for image normalization.
        normalization_std (List[float]): Std values for image normalization.

    Method __init__(encoder, prompt_encoder, decoder, normalization_mean, normalization_std):
        Initialize SegmentModel with encoder, prompt_encoder, and decoder.
        Set normalization_mean and normalization_std for image normalization.

    Method preprocess_image(image):
        Normalize the input image using normalization_mean and normalization_std.
        Return the normalized image.

    Method forward(image, prompt):
        Normalize the input image using preprocess_image method.
        Generate image embeddings by passing normalized image through encoder.
        Encode the input prompt using prompt_encoder.
        Combine image embeddings and prompt embeddings.
        Decode combined embeddings to predict masks using decoder.
        Apply mask_threshold to determine valid masks.
        Return the predicted masks.

"""
