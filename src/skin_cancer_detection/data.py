import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import load_img, img_to_array
from typing import Tuple, Dict, Any

from . import config
from .logger import logger

def load_metadata() -> pd.DataFrame:
    """
    Loads metadata from CSV.

    Returns:
        pd.DataFrame: Loaded metadata dataframe.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    if not os.path.exists(config.METADATA_PATH):
        logger.error(f"Metadata file not found at {config.METADATA_PATH}")
        raise FileNotFoundError(f"Metadata file not found at {config.METADATA_PATH}")
    
    logger.info(f"Loading metadata from {config.METADATA_PATH}")
    return pd.read_csv(config.METADATA_PATH)

def preprocess_metadata(metadata: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encodes diagnosis labels and adds them to metadata.

    Args:
        metadata (pd.DataFrame): Raw metadata.
    
    Returns:
        Tuple[pd.DataFrame, LabelEncoder]: Metadata with 'label' column and the fitted LabelEncoder.
    """
    logger.info("Preprocessing metadata and encoding labels...")
    le = LabelEncoder()
    metadata['label'] = le.fit_transform(metadata['diagnosis'])
    return metadata, le

def load_and_preprocess_image(img_id: tf.Tensor, label: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses a single image.
    Intended to be wrapped in a tf.py_function.
    
    Args:
        img_id (tf.Tensor): Tensor containing the image ID string.
        label (tf.Tensor): Tensor containing the label.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed image array and label.
    """
    try:
        img_id_str = img_id.numpy().decode('utf-8')
        img_path = os.path.join(config.IMAGE_DIR, img_id_str + '.jpg')
        
        image = load_img(img_path, target_size=config.IMG_SIZE)
        image = img_to_array(image)
        image = image / 255.0  # Normalize to [0,1]
    except Exception as e:
        # TF data pipeline relies on this functioning correctly.
        # Returning zeros might be a safe fallback or letting it crash depends on design.
        # For now, we log exception but we must return something matching the signature.
        # ideally we should filter these out beforehand.
        logger.warning(f"Error loading image {img_id}: {e}")
        image = np.zeros(config.IMG_SIZE + (3,), dtype=np.float32)
    
    return image, label

def load_and_preprocess_image_tf(img_id: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Wrapper for TensorFlow dataset mapping.

    Args:
        img_id (tf.Tensor): Image ID.
        label (tf.Tensor): Label.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Typed image and label tensors.
    """
    image, label = tf.py_function(
        func=load_and_preprocess_image, 
        inp=[img_id, label], 
        Tout=[tf.float32, tf.int32]
    )
    image.set_shape(config.IMG_SIZE + (3,))
    label.set_shape([])
    return image, label

def create_dataset(metadata: pd.DataFrame) -> tf.data.Dataset:
    """
    Creates a basic tf.data.Dataset from metadata.
    
    Args:
        metadata (pd.DataFrame): Metadata with 'isic_id' and 'label'.

    Returns:
        tf.data.Dataset: The created dataset.
    """
    logger.info("Creating TensorFlow dataset...")
    image_labels_ds = tf.data.Dataset.from_tensor_slices((
        metadata['isic_id'].values,
        metadata['label'].values.astype(np.int32)
    ))
    
    dataset = image_labels_ds.map(
        load_and_preprocess_image_tf, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset

def get_data_augmentation_layer() -> tf.keras.Sequential:
    """
    Returns the data augmentation sequential model.

    Returns:
        tf.keras.Sequential: Data augmentation layers.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
    ])

def prepare_datasets(metadata: pd.DataFrame, dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Splits dataset into train, val, test and applies batching/prefetching.

    Args:
        metadata (pd.DataFrame): Metadata for calculating sizes.
        dataset (tf.data.Dataset): The base dataset.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    logger.info("Splitting dataset into train, validation, and test sets...")
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

def calculate_class_weights(metadata: pd.DataFrame) -> Dict[int, float]:
    """
    Calculates class weights to handle imbalance.
    
    Args:
        metadata (pd.DataFrame): Metadata with 'label'.

    Returns:
        Dict[int, float]: Dictionary mapping class indices to weights.
    """
    logger.info("Calculating class weights...")
    all_labels = metadata['label'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    logger.info(f"Class weights: {weights_dict}")
    return weights_dict
