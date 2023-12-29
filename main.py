from fastapi import FastAPI, File, UploadFile
from typing import List, Dict, Union
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import tensorflow as tf
import uvicorn
import io
from PIL import Image

app = FastAPI()

loaded_models = {}

def load_keras_model(model_folder):
    model = keras.models.load_model(model_folder)
    return model

async def classify_image(model, file):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    image_array = np.asarray(img.resize((224, 224)))[..., :3]
    image_array = np.copy(image_array)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    classes = ['aedes', 'culex']
    class_name = classes[np.argmax(score)]
    confidence_score = float(score[np.argmax(score)])
    print(f"Score: {score}")

    return {"result": class_name, "score": confidence_score}

@app.on_event("startup")
async def load_models():
    model_folder = "./models/"
    
    for model_type in os.listdir(model_folder):
        model_path = os.path.join(model_folder, model_type)
        loaded_models[model_type] = load_keras_model(model_path)
        print(f"Loaded model for {model_type}")    

def perform_classification(model, file: UploadFile):

    result = classify_image(model, file)
    
    return result

@app.post("/classification")
async def classification(
    abdomen: UploadFile = None,
    kepala: UploadFile = None,
    siphon: UploadFile = None,
    fullbody: UploadFile = None
):
    files = [abdomen, kepala, siphon, fullbody]

    results: Dict[str, Dict[str, float]] = {}

    for file, image_type in zip(files, ["abdomen", "kepala", "siphon", "fullbody"]):
        if file is not None and image_type in loaded_models:
            model = loaded_models[image_type]
            result = await perform_classification(model, file)
            results[image_type] = result
        else:
            results[image_type] = {"result": "Not received image", "score": 0.0}

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)