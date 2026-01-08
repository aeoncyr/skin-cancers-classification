# Skin Cancer Detection Project

This project is a machine learning application aimed at detecting different types of skin cancer using image data. It assumes the HAM10000 dataset is present.

## 🚀 Features

- **Modular Architecture:** Cleanly separated concerns (Data, Model, Training, Visualization) in a Python package structure.
- **Deep Learning Model:** Uses **MobileNetV2** with Transfer Learning for efficient and accurate classification.
- **Robust Training Pipeline:** Includes Data Augmentation, Class Weighting (to handle imbalance), Early Stopping, and Learning Rate Scheduling.
- **Educational Resources:** A dedicated `Learning/` module with interactive notebooks explaining the Medical, Mathematical, and Statistical concepts behind the project.
- **Professional Engineering:** integrated logging, type hinting, and CLI arguments.

## 📂 Project Structure

```
├── dataset/                  # HAM10000 dataset
├── Learning/                 # Educational Interactive Notebooks
│   ├── 01_Medical_Context_HAM10000.ipynb
│   ├── 02_MobileNetV2_Architecture.ipynb
│   └── 03_Medical_AI_Evaluation.ipynb
├── notebooks/                # Demonstration Notebooks
│   ├── dataUnderstanding.ipynb
│   └── modelTraining.ipynb
├── scripts/                  # CLI Entry Points
│   ├── run_analysis.py
│   └── train_model.py
├── src/                      # Source Code Package
│   └── skin_cancer_detection/
│       ├── config.py
│       ├── data.py
│       ├── logger.py
│       ├── model.py
│       ├── train.py
│       └── visualization.py
├── requirements.txt
└── README.md
```

## 🎓 Learning Module

New to Medical AI? Check out the `Learning/` folder:
1. **Medical Context:** Understand the pathology of Melanoma vs. Nevi and the ABCD Rule.
2. **Architecture:** Learn why we use Depthwise Separable Convolutions (MobileNetV2).
3. **Evaluation:** Why "Accuracy" is dangerous in medicine, and how to use Sensitivity/Specificity.

## 🛠️ Usage

### 1. Installation

```bash
git clone https://github.com/aeoncyr/skin-cancers-classification.git
pip install -r requirements.txt
```

### 2. Run Data Analysis

Generate visualizations for Class Distribution, Sample Images, and Correlations.

```bash
python scripts/run_analysis.py
```

### 3. Train the Model

You can train the model using the default settings or override them via CLI arguments.

**Default:**
```bash
python scripts/train_model.py
```

**Custom Training:**
```bash
python scripts/train_model.py --epochs 50 --batch-size 16 --learning-rate 0.001 --save-path "my_model.h5"
```

### 4. Notebooks

For an interactive experience, launch Jupyter Lab or Notebook:

- **`notebooks/dataUnderstanding.ipynb`**: Step-by-step dataset exploration.
- **`notebooks/modelTraining.ipynb`**: Interactive training demo.

## 📊 Dataset

The project uses the **HAM10000** dataset. Ensure the dataset is placed in:
- Images: `dataset/ham10000/*.jpg`
- Metadata: `dataset/ham10000_metadata.csv`

## 📄 License

CC0 1.0 Universal
