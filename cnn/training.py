import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks

class VehicleClassifierCNN:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_names = ['auto rickshaw', 'bycycle', 'car', 'cng', 'motor bike', 'taxi', 'truck']
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.model = self.create_model()

    def load_data(self):
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    self.images.append(img)
                    self.labels.append(class_name)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def encode_labels(self):
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

    def create_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels_encoded, test_size=0.2, random_state=42, stratify=self.labels_encoded)
        # Data normalization
        X_train_normalized = X_train / 255.0
        X_test_normalized = X_test / 255.0
        # Training callbacks
        early_stopping = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        # Train the model
        self.model.fit(X_train_normalized, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping])
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_test_normalized, y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        # Save the model in .h5 format
        self.model.save('cnn_model.h5')

    def load_model(self, model_path='cnn_model.h5'):
        self.model = models.load_model(model_path)
        self.label_encoder = LabelEncoder()
        # Load label encoder from file (if exists)
        labels_file = os.path.splitext(model_path)[0] + '_label_encoder.pkl'
        if os.path.exists(labels_file):
            self.label_encoder = joblib.load(labels_file)

    def predict(self, image):
        img_resized = cv2.resize(image, (64, 64))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        prediction = self.model.predict(img_expanded)
        predicted_class_index = np.argmax(prediction)
        return self.label_encoder.inverse_transform([predicted_class_index])[0]

# Example usage
if __name__ == "__main__":
    classifier_cnn = VehicleClassifierCNN('../Car')
    classifier_cnn.load_data()
    classifier_cnn.encode_labels()
    classifier_cnn.train_model()
