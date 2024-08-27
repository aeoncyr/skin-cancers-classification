# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

# Set the paths
image_dir = 'dataset/ham10000/'
metadata_path = 'dataset/ham10000_metadata.csv'

# Load the metadata
metadata = pd.read_csv(metadata_path)

# Encode the diagnosis labels into integers
le = LabelEncoder()
metadata['label'] = le.fit_transform(metadata['diagnosis'])

# Number of classes (unique diagnoses)
num_classes = len(metadata['label'].unique())

# Define a function to load and preprocess each image
def load_and_preprocess_image(img_id, label, img_size=(128, 128)):
    img_id_str = img_id.numpy().decode('utf-8')
    img_path = os.path.join(image_dir, img_id_str + '.jpg')
    image = load_img(img_path, target_size=img_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0,1]
    return image, label

# Adjust the function to handle TensorFlow tensors properly
def load_and_preprocess_image_tf(img_id, label):
    image, label = tf.py_function(
        func=load_and_preprocess_image, 
        inp=[img_id, label], 
        Tout=[tf.float32, tf.int32]  # Ensure the label is returned as int32
    )
    image.set_shape((128, 128, 3))
    label.set_shape([])
    return image, label

# Create a TensorFlow Dataset from the metadata
image_labels_ds = tf.data.Dataset.from_tensor_slices((
    metadata['isic_id'].values,
    metadata['label'].values.astype(np.int32)  # Cast labels to int32
))

# Map the dataset with the correct function
dataset = image_labels_ds.map(load_and_preprocess_image_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Data augmentation using tf.keras.layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomTranslation(0.2, 0.2),
])

# Apply data augmentation to the training dataset
augmented_train_dataset = dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# Batch and prefetch the dataset for training
batch_size = 32
train_size = int(0.8 * len(metadata))
val_size = int(0.1 * len(metadata))

train_dataset = augmented_train_dataset.take(train_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = dataset.skip(train_size).take(val_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = dataset.skip(train_size + val_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Build the CNN model using Keras layers with MobileNetV2 Backbone
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight

# Collect all labels from the training dataset
all_labels = []

for _, y_batch in train_dataset:
    all_labels.extend(y_batch.numpy())

# Calculate the class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

# Create a dictionary mapping class indices to weights
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}


# Load the base model with pre-trained ImageNet weights
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers on top of the base model
def create_model(learning_rate=0.01, dropout_rate=0.0):
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=1)

# Define the grid search parameters
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.3, 0.5, 0.7],
    'batch_size': [32, 64, 128]
}

# Search for the best parameters
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10)
grid_result = grid.fit(train_dataset, validation_data=val_dataset)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[early_stopping, lr_scheduler],
    class_weights=class_weight_dict,
    verbose=2
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.2f}")

# Save the model
model.save("/content/drive/MyDrive/skin_cancer_cnn_model.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# To predict and display the output probabilities
sample_image = next(iter(test_dataset.take(1)))[0]
predictions = model.predict(sample_image)
predicted_class = np.argmax(predictions[0])
predicted_label = le.inverse_transform([predicted_class])[0]
confidence = np.max(predictions) * 100

print(f"Prediction: {confidence:.2f}% similar to {predicted_label}")
