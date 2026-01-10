import os
import sys
import io
import logging
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path so `import src.*` works when uvicorn is started
# from the repo root or from the api folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Now safe to import project modules
try:
    from src.model import SimpleCNN
    from src.preprocess import preprocess_image
except Exception as e:
    # Import errors will be surfaced in /health and when trying to predict
    SimpleCNN = None
    preprocess_image = None
    import_error = str(e)
else:
    import_error = None

# Config
MODEL_REL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "saved_model", "steel_defect_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORS_ALLOWED = os.environ.get("CORS_ALLOWED_ORIGINS", "*")

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("steel-defect-api")

# Class names (must match training order)
CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

app = FastAPI(title="Steel Surface Defect Classification API")

# Configure CORS
if CORS_ALLOWED == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in CORS_ALLOWED.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


# Load model function
def load_model(path: str):
    if SimpleCNN is None:
        raise ImportError(f"Project imports failed: {import_error}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'")
    model = SimpleCNN().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# Try load on startup
try:
    model = load_model(MODEL_REL_PATH)
    load_error = None
    logger.info(f"Model loaded from {MODEL_REL_PATH} on device {DEVICE}")
except Exception as e:
    model = None
    load_error = str(e)
    logger.error(f"Model load failed: {load_error}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "load_error": load_error}


@app.get("/classes", response_model=List[str])
def get_classes():
    return CLASSES


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts a multipart/form-data image upload and returns:
    - prediction: top class name
    - confidence: probability of top class
    - probabilities: dict of class -> probability
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if preprocess_image is None:
        raise HTTPException(status_code=500, detail=f"Preprocess import failed: {import_error}")

    try:
        input_tensor = preprocess_image(image).to(DEVICE)  # expected shape (1,1,224,224)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    with torch.no_grad():
        outputs = model(input_tensor)  # shape (1, num_classes)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    prediction = CLASSES[top_idx]
    confidence = float(probs[top_idx])
    probabilities = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    return PredictionResponse(prediction=prediction, confidence=confidence, probabilities=probabilities)