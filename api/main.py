from fastapi import FastAPI,UploadFile,File
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# MODEL = tf.keras.models.load_model("C:/My Programs/Machine Learning Projects/potato-disease-classification/saved_models/1")
# MODEL = TFSMLayer("C:/My Programs/Machine Learning Projects/Potato Disease Detection/saved_models/1", call_endpoint="serving_default")
MODEL=tf.keras.models.load_model("C:/My Programs/Machine Learning Projects/potato-disease-classification/saved_models/2.keras")


CLASS_NAMES=["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello,Gaurav"

def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    

    # ChatGPT
    # image = Image.open(BytesIO(data)).convert("RGB")
    # image = image.resize((256, 256))  # Or use your model's input size
    # image = np.array(image).astype(np.float32) / 255.0  # Convert to float32 and normalize
    return image

@app.post("/predict")
async def predict(
    file:UploadFile=File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

        
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8001)