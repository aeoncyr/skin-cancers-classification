# Skin Cancer Detection Project

This project is a machine learning application aimed at detecting different types of skin cancer using image data. It assumes the HAM10000 dataset is present.

## Project Structure

The project has been restructured for better maintainability:

- `src/skin_cancer_detection/`: The main Python package containing the source code.
    - `config.py`: Configuration constants.
    - `data.py`: Data loading and preprocessing logic.
    - `model.py`: Model architecture definition (CNN with MobileNetV2).
    - `train.py`: Training loop and evaluation.
    - `visualization.py`: Functions for data exploration and result plotting.
- `scripts/`: Entry points for running analysis and training.
    - `run_analysis.py`: Runs data understanding visualizations.
    - `train_model.py`: Trains the model.
- `notebooks/`: Jupyter notebooks for experimentation.
    - `dataUnderstanding.ipynb`: Original data analysis notebook.
    - `modelTraining.ipynb`: Original model training notebook.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: Ensure you have `tensorflow`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` installed if not in `requirements.txt`)

## Usage

### 1. Data Understanding / Analysis

To generate visualizations and explore the dataset:

```bash
python scripts/run_analysis.py
```

### 2. Model Training

To train the skin cancer detection model:

```bash
python scripts/train_model.py
```

The model will be saved as `skin_cancer_cnn_model.h5`.

## Dataset

The project uses the **HAM10000** dataset. Ensure the dataset is placed in the `dataset/ham10000` directory and metadata in `dataset/ham10000_metadata.csv` relative to the project root, or update `src/skin_cancer_detection/config.py`.

## License

CC0 1.0 Universal
