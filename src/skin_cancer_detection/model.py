import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from . import config
from .logger import logger

def create_base_model() -> tf.keras.Model:
    """
    Creates the base MobileNetV2 model.

    Returns:
        tf.keras.Model: Pre-trained MobileNetV2 model (frozen).
    """
    logger.info("Initializing MobileNetV2 base model...")
    base_model = MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False
    return base_model

def create_model(learning_rate: float = config.LEARNING_RATE, dropout_rate: float = config.DROPOUT_RATE, num_classes: int = config.NUM_CLASSES) -> tf.keras.Model:
    """
    Creates and compiles the CNN model.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate.
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    logger.info(f"Building model with LR={learning_rate}, Dropout={dropout_rate}, Classes={num_classes}")
    base_model = create_base_model()
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    logger.info("Model compiled successfully.")
    return model
