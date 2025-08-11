from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json
import os

app = FastAPI(title="Plant Classifier API")

MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_classifier.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

#load class names 
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

print(f"Loaded model with input shape: {model.input_shape}")
print(f"Loaded class names: {CLASS_NAMES}")

@app.get("/")
def root():
    return {"message": "Plant Classifier API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
    
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        input_height, input_width = model.input_shape[1], model.input_shape[2]
        image = image.resize((input_width, input_height))

        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

