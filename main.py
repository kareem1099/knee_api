from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
from torchvision import models
from io import BytesIO
from utils.download_knee_model import download_knee_model
import uvicorn


app = FastAPI(title="Knee Osteoarthritis Predictor")

# Load model
knee_model_path = download_knee_model()
knee_model = models.resnet50(weights=None)
knee_model.fc = torch.nn.Linear(knee_model.fc.in_features, 2)
checkpoint = torch.load(knee_model_path, map_location=torch.device("cpu"))
knee_model.load_state_dict(checkpoint)
knee_model.eval()

def preprocess_knee_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.transpose(image_array, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean[:, None, None]) / std[:, None, None]
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
    return image_tensor

@app.post("/predict_Knee")
async def predict_knee(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_knee_image(image)
        with torch.no_grad():
            prediction = knee_model(processed_image)
        predicted_class = torch.argmax(prediction, dim=1).item()
        result = "Healthy knee" if predicted_class == 0 else "There is knee Osteoarthritis"
        return {"result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
