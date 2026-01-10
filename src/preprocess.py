# src/preprocess.py

from PIL import Image
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()   # ONLY this
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # (1, 1, 224, 224)
    return image_tensor