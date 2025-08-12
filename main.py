from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "model\model.keras"
model = load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) 
    return np.expand_dims(img_array, axis=0)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("templates/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    prediction = model.predict(img_array)[0][0]  # Assuming binary classification
    label = "Autistic" if prediction < 0.5 else "Non-Autistic"
    return {"prediction": label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

