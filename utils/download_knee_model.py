import os
import gdown

MODEL_DIR = "models"
MODEL_NAME = "best_knee_model_resnet5.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

GOOGLE_DRIVE_ID = "1vdugidpz6qFym32QfgHzzLvxNQzBbCjy"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

def download_knee_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading Knee model...")
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
        print("Knee model download complete.")
    else:
        print("Knee model already exists.")
    return MODEL_PATH
