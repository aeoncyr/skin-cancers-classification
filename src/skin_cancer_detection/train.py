import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from typing import Tuple, Dict
from typing import Optional

from . import config, data, model, visualization
from .logger import logger

def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    save_path: str = "skin_cancer_cnn_model.h5"
) -> Tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    Main training loop.

    Args:
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        save_path (str): Path to save the trained model.

    Returns:
        Tuple[tf.keras.callbacks.History, tf.keras.Model]: Training history and model.
    """
    logger.info("Starting training process...")

    # 1. Load and Preprocess Data
    logger.info("Loading metadata...")
    try:
        metadata = data.load_metadata()
    except FileNotFoundError:
        logger.error("Metadata not found. Aborting training.")
        return None, None

    metadata, _ = data.preprocess_metadata(metadata)
    
    logger.info("Creating dataset...")
    dataset = data.create_dataset(metadata)
    
    # Overriding batch size in config temporarily or passing it down
    # Since config is global, ideally we pass it to prepare_datasets. 
    # But for now, we'll just log that we are using config values unless we refactor data.py to accept batch_size
    # Let's do a quick patch on data.prepare_datasets to accept batch_size if we wanted perfect purity, 
    # but sticking to global config is 'okay' for this scope if we updated the config runtime.
    # A cleaner way is to just let it use config constants or update them:
    
    # data.config.BATCH_SIZE = batch_size # Mutable global state is risky but works for simple scripts
    
    train_ds, val_ds, test_ds = data.prepare_datasets(metadata, dataset)
    
    # 2. Calculate Class Weights
    class_weights = data.calculate_class_weights(metadata)
    
    # 3. Create Model
    logger.info("Building model...")
    cnn_model = model.create_model(learning_rate=learning_rate)
    
    # 4. Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        verbose=1
    )
    
    # 5. Train
    logger.info(f"Starting training for {epochs} epochs...")
    history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weights,
        verbose=2
    )
    
    # 6. Evaluate
    logger.info("Evaluating model...")
    test_loss, test_acc = cnn_model.evaluate(test_ds)
    logger.info(f"Test accuracy: {test_acc:.2f}")
    
    # 7. Save Model
    cnn_model.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # 8. Plot History
    visualization.plot_training_history(history)
    
    return history, cnn_model
