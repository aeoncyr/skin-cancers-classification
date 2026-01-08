import sys
import os
import argparse

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from skin_cancer_detection import train
from skin_cancer_detection.logger import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train Skin Cancer Detection Model")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save-path', type=str, default='skin_cancer_cnn_model.h5', help='Path to save model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting training with args: {vars(args)}")
    
    # Note: data.py uses config.BATCH_SIZE global. 
    # If we want to support dynamic batch size without extensive refactor, 
    # we can inject it into the config or refactor data.py.
    # For now, let's keep it simple and just acknowledge it. To strictly support CLI batch size, 
    # we would need to update `src/skin_cancer_detection/config.py` in runtime or refactor.
    # Let's do a runtime patch for now as it's efficient.
    from skin_cancer_detection import config
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    
    try:
        train.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
