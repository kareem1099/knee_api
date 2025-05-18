from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from io import BytesIO
import os
import uvicorn

app = FastAPI(title="Knee Osteoarthritis Predictor")

# Load pre-trained model from file
model_path = "model.pt"  # Replace with your traced model path
knee_model = torch.jit.load(model_path, map_location=torch.device("cpu"))
knee_model.eval()

# Minimal preprocessing with only PIL and Torch
def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.resize((224, 224))
    image = image.convert("RGB")
    img_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    img_tensor = img_tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0)

@app.post("/predict_Knee")
async def predict_knee(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        tensor = preprocess_image(image)
        with torch.no_grad():
            output = knee_model(tensor)
        pred = torch.argmax(output, dim=1).item()
        result = "Healthy knee" if pred == 0 else "There is knee Osteoarthritis"
        return {"result": result}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
