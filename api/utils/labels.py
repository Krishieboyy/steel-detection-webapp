from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
CLASS_TO_IDX_PATH = BASE_DIR / "class_to_idx.json"

with open(CLASS_TO_IDX_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {int(v): k for k, v in class_to_idx.items()}
CLASSES = [idx_to_class[i] for i in range(len(idx_to_class))]