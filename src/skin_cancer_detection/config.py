import os
from typing import Tuple

# Paths
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR: str = os.path.join(BASE_DIR, 'dataset')
IMAGE_DIR: str = os.path.join(DATA_DIR, 'ham10000')
METADATA_PATH: str = os.path.join(DATA_DIR, 'ham10000_metadata.csv')

# Model Parameters
IMG_HEIGHT: int = 128
IMG_WIDTH: int = 128
IMG_SIZE: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE: int = 32
NUM_CLASSES: int = 7 # Based on ham10000 dataset

# Training Parameters
EPOCHS: int = 20
LEARNING_RATE: float = 0.01
DROPOUT_RATE: float = 0.5
