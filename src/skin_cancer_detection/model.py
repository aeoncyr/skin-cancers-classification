import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from . import config

def create_base_model():
    """Creates the base MobileNetV2 model."""
    base_model = MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False
    return base_model

def create_model(learning_rate=config.LEARNING_RATE, dropout_rate=config.DROPOUT_RATE, num_classes=config.NUM_CLASSES):
    """
    Creates and compiles the CNN model.
    """
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
    return model
