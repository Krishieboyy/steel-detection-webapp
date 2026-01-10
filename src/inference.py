import torch
from torchvision import transforms
from PIL import Image
import sys
import os

from src.model import SimpleCNN

# class names (same order as training)
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = SimpleCNN().to(device)
model_path = os.path.join("saved_model", "steel_defect_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# preprocessing (EXACT same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return CLASSES[predicted.item()]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    prediction = predict(image_path)
    print(f"Predicted defect: {prediction}")
