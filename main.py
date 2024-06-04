import enum
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from vgg16.predict import VGG16Predict

app = FastAPI()

vgg16 = VGG16Predict()

class ModelOption(enum.Enum):
      knn = 'knn'
      svm = 'svm'
      vgg16 = 'vgg16'
      random_forest = 'random_forest'

# Function to extract HOG features
def extract_hog_features(image):
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      hog_features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
      return hog_features

@app.post("/predict")
async def predict(model_name:ModelOption, file: UploadFile = File(...)):
      contents = await file.read()
      if model_name.value == "vgg16":
            result = vgg16.predict_image(contents)
            class_name = result.get("class")
      else:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Resize the image to the same size used during training
            img_resized = cv2.resize(img, (64, 64))
            
            # Extract HOG features
            features = extract_hog_features(img_resized)
            features = features.reshape(1, -1)  # Reshape for prediction
            model = joblib.load(f'{model_name.value}/{model_name.value}_model.pkl')
            label_encoder = joblib.load(f'{model_name.value}/label_encoder.pkl')

            # Predict the class of the image
            prediction = model.predict(features)
            class_name = label_encoder.inverse_transform(prediction)[0]
      return JSONResponse(content={"class_name": class_name})
