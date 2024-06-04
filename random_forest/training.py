import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

class VehicleClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_names = ['auto rickshaw', 'bycycle', 'car', 'cng', 'motor bike', 'taxi', 'truck']
        self.images = []
        self.labels = []
        self.features = None
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

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

    def extract_features(self):
        self.features = np.array([self.extract_hog_features(image) for image in self.images])
    
    @staticmethod
    def extract_hog_features(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        return hog_features

    def encode_labels(self):
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels_encoded, test_size=0.2, random_state=42, stratify=self.labels_encoded)  # Menambahkan stratify untuk memastikan pembagian yang seimbang pada kelas
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        joblib.dump(self.model, 'random_forest_model.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')

    def load_model(self, model_path='random_forest_model.pkl', label_encoder_path='label_encoder.pkl'):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def predict(self, image):
        img_resized = cv2.resize(image, (64, 64))
        features = self.extract_hog_features(img_resized).reshape(1, -1)
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]

# Example usage
if __name__ == "__main__":
    classifier = VehicleClassifier('../Car')
    classifier.load_data()
    classifier.extract_features()
    classifier.encode_labels()
    classifier.train_model()
