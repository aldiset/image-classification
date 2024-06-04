# Import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define data path (replace with your actual path)
data_dir = '../Car'  # Path to your dataset folder

# Set image dimensions
img_width, img_height = 150, 150  # Adjust these values as needed

# Create data generator
train_datagen = ImageDataGenerator(rescale=1./255,  # Normalize pixel values
                                   shear_range=0.2,  # Randomly shear images
                                   zoom_range=0.2,  # Randomly zoom images
                                   horizontal_flip=True)  # Randomly flip images horizontally

# Load training data with class_mode='categorical' for multi-class classification
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'  # Multi-class classification
)

# Get number of classes from the class_indices attribute
num_classes = len(train_generator.class_indices)

# Create the model (transfer learning with VGG16)
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False  # Freeze the base model layers for transfer learning

# Add custom layers on top of the pre-trained base
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)  # Dense layer with ReLU activation
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Output layer with softmax activation for multi-class classification

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=10,  # Adjust the number of epochs as needed
                    verbose=2)  # Set verbose level to 2 for detailed training progress

# Evaluate the model (optional)
# Since there's no validation set, evaluation on training data can give an indication of performance
loss, accuracy = model.evaluate(train_generator)
print('Training Loss:', loss)
print('Training Accuracy:', accuracy)

# Save the model (optional)
# model.save('car_classifier.h5')