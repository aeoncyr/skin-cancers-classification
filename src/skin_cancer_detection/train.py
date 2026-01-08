import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from . import config, data, model, visualization

def train():
    """
    Main training loop.
    """
    # 1. Load and Preprocess Data
    print("Loading metadata...")
    metadata = data.load_metadata()
    metadata, _ = data.preprocess_metadata(metadata)
    
    print("Creating dataset...")
    dataset = data.create_dataset(metadata)
    train_ds, val_ds, test_ds = data.prepare_datasets(metadata, dataset)
    
    # 2. Calculate Class Weights
    class_weights = data.calculate_class_weights(metadata)
    
    # 3. Create Model
    print("Building model...")
    cnn_model = model.create_model()
    
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
    print("Starting training...")
    history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weights,
        verbose=2
    )
    
    # 6. Evaluate
    print("Evaluating model...")
    test_loss, test_acc = cnn_model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.2f}")
    
    # 7. Save Model
    save_path = "skin_cancer_cnn_model.h5"
    cnn_model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # 8. Plot History (Optional)
    # in a script environment, this might show validation window or save to file
    # for now we call it, user can see if running interactively
    visualization.plot_training_history(history)
    
    return history, cnn_model
