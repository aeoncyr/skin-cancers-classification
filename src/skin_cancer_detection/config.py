import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
IMAGE_DIR = os.path.join(DATA_DIR, 'ham10000')
METADATA_PATH = os.path.join(DATA_DIR, 'ham10000_metadata.csv')

# Model Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
NUM_CLASSES = 7 # Based on ham10000 dataset

# Training Parameters
EPOCHS = 20
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.5
