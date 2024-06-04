import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from skimage.feature import hog

class VehicleClassifierKNN:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_names = ['auto rickshaw', 'bycycle', 'car', 'cng', 'motor bike', 'taxi', 'truck']
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.model = KNeighborsClassifier(n_neighbors=3)

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

    def extract_hog_features(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        return hog_features

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels_encoded, test_size=0.2, random_state=42, stratify=self.labels_encoded)
        X_train_features = np.array([self.extract_hog_features(img) for img in X_train])
        X_test_features = np.array([self.extract_hog_features(img) for img in X_test])
        self.model.fit(X_train_features, y_train)
        y_pred = self.model.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy}")
        # Save the model and label encoder
        joblib.dump(self.model, 'knn_model.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')

    def load_model(self, model_path='knn_model.pkl', label_encoder_path='label_encoder.pkl'):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        # Ensure label encoder is fitted
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("LabelEncoder is not fitted yet. Please fit LabelEncoder before loading the model.")

    def predict(self, image):
        img_resized = cv2.resize(image, (64, 64))
        img_features = self.extract_hog_features(img_resized)
        img_features = img_features.reshape(1, -1)  # Reshape for prediction
        prediction = self.model.predict(img_features)
        return self.label_encoder.inverse_transform(prediction)[0]

# Example usage
if __name__ == "__main__":
    classifier_knn = VehicleClassifierKNN('../Car')
    classifier_knn.load_data()
    classifier_knn.encode_labels()
    classifier_knn.train_model()
