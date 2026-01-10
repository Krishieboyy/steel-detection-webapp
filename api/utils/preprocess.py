# src/preprocess.py

from PIL import Image
import torchvision.transforms as transforms
import torch

# Image preprocessing pipeline
# MUST match training preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # because training used grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.

    Args:
        image (PIL.Image): Input image uploaded via API

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 1, 224, 224)
    """

    # Ensure image is in RGB before converting to grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_tensor = transform(image)

    # Add batch dimension: (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor