import sys
import os

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from skin_cancer_detection import train

if __name__ == "__main__":
    train.train()
