import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import load_img, img_to_array
from . import config

def load_metadata():
    """Loads metadata from CSV."""
    if not os.path.exists(config.METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {config.METADATA_PATH}")
    return pd.read_csv(config.METADATA_PATH)

def preprocess_metadata(metadata):
    """
    Encodes diagnosis labels and adds them to metadata.
    Returns metadata and the label encoder.
    """
    le = LabelEncoder()
    metadata['label'] = le.fit_transform(metadata['diagnosis'])
    return metadata, le

def load_and_preprocess_image(img_id, label):
    """
    Loads and preprocesses a single image.
    Intended to be wrapped in a tf.py_function.
    """
    img_id_str = img_id.numpy().decode('utf-8')
    img_path = os.path.join(config.IMAGE_DIR, img_id_str + '.jpg')
    
    # Robust check or just let it fail/handle?
    # TF data pipeline might prefer it to fail specific way or supply empty.
    # For now, following original logic but using config constants.
    
    image = load_img(img_path, target_size=config.IMG_SIZE)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0,1]
    
    return image, label

def load_and_preprocess_image_tf(img_id, label):
    """Wrapper for TensorFlow dataset mapping."""
    image, label = tf.py_function(
        func=load_and_preprocess_image, 
        inp=[img_id, label], 
        Tout=[tf.float32, tf.int32]
    )
    image.set_shape(config.IMG_SIZE + (3,))
    label.set_shape([])
    return image, label

def create_dataset(metadata):
    """Creates a basic tf.data.Dataset from metadata."""
    image_labels_ds = tf.data.Dataset.from_tensor_slices((
        metadata['isic_id'].values,
        metadata['label'].values.astype(np.int32)
    ))
    
    dataset = image_labels_ds.map(
        load_and_preprocess_image_tf, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset

def get_data_augmentation_layer():
    """Returns the data augmentation sequential model."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
    ])

def prepare_datasets(metadata, dataset):
    """
    Splits dataset into train, val, test and applies batching/prefetching.
    """
    train_size = int(0.8 * len(metadata))
    val_size = int(0.1 * len(metadata))
    
    augmented_train_dataset = dataset.map(
        lambda x, y: (get_data_augmentation_layer()(x, training=True), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    train_dataset = augmented_train_dataset.take(train_size).batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.skip(train_size).take(val_size).batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = dataset.skip(train_size + val_size).batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def calculate_class_weights(metadata):
    """Calculates class weights to handle imbalance."""
    # Assuming 'label' column exists
    all_labels = metadata['label'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    return {i: weight for i, weight in enumerate(class_weights)}
