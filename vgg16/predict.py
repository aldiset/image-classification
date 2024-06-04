import io
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

class VGG16Predict:
    """
    A class for image classification using a pre-trained VGG16 model.
    """

    def __init__(self, model_path='vgg16/car_classifier.h5'):
        """
        Loads the pre-trained model.

        Args:
            model_path (str, optional): Path to the saved model file.
                Defaults to 'vgg16/car_classifier.h5'.
        """
        self.model = load_model(model_path)

    def preprocess_image(self, image: bytes):
        """
        Preprocesses an image for classification.

        Args:
            image (bytes): Bytes representing the image content.

        Returns:
            A preprocessed NumPy array representing the image.
        """

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image))

        # Resize the image
        image = image.resize((150, 150))

        # Convert to NumPy array
        image_array = np.array(image)

        # Expand the dimension (add a new channel dimension)
        image_array = np.expand_dims(image_array, axis=0)

        # Normalize the pixel values
        image_array = image_array.astype('float32') / 255.0

        return image_array

    def predict_image(self, image: bytes):
        """
        Predicts the class of a preprocessed image.

        Args:
            image (bytes): Bytes representing the preprocessed image.

        Returns:
            A dictionary containing the predicted class and confidence score.
        """

        # Preprocess the image
        preprocessed_image = self.preprocess_image(image)

        # Make the prediction using the model
        prediction = self.model.predict(preprocessed_image)[0]

        # Get the class label and confidence score
        class_label = np.argmax(prediction)
        confidence_score = np.max(prediction)

        # Convert class label to a human-readable label
        class_names = {
            0: "Auto Rickshaw",
            1: "Bicycle",
            2: "Car",
            3: "CNG",
            4: "Motor Bike",
            5: "Taxi",
            6: "Truck",
        }
        class_label = class_names.get(class_label, "Unknown")

        # Return the classification result
        return {
            "class": class_label,
            "confidence": confidence_score
        }

# Example usage:
# predictor = VGG16Predict()
# result = predictor.predict_image(image_bytes)
# print(result)
