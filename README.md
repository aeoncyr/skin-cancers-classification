# Skin Cancer Detection Project -- On progress
This is a pilot project for research on skin cancer risk factors assessment application. The model is trained using the HAM-10000 dataset from ISIC by Philipp Tschandl.

This project is a machine learning application aimed at detecting different types of skin cancer using image data. The model is trained on the HAM10000 dataset, a collection of dermatoscopic images of common pigmented skin lesions. It leverages a Convolutional Neural Network (CNN) to classify skin lesions into various diagnostic categories.

## Project Overview

- **Data Understanding:** The `dataUnderstanding.py` script performs initial data exploration, visualization, and analysis of the HAM10000 dataset, helping to understand the distribution of skin cancer types.
  
- **Model Training:** The `modelTraining.py` script handles the preprocessing of images, data augmentation, model architecture definition, and training of a CNN model to classify skin cancer lesions.

## Features

- **Data Augmentation:** Random flips, rotations, and zooms are applied to the images to enhance model generalization.
- **Model Architecture:** A CNN built with TensorFlow/Keras, including layers like Convolution, MaxPooling, Dropout, and Dense layers for classification.
- **Early Stopping & Learning Rate Scheduler:** These callbacks help improve training efficiency by avoiding overfitting and adjusting the learning rate dynamically.
- **Evaluation & Visualization:** The training history and performance metrics are visualized to assess model accuracy and loss over epochs.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/aeoncyr/skinCancerDetection.git
   cd skinCancerDetection
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses the **HAM10000** dataset, which contains dermatoscopic images of various types of skin lesions. The dataset includes the following lesion types:
- Melanocytic nevi
- Melanoma
- Benign keratosis-like lesions
- Basal cell carcinoma
- Actinic keratoses
- Vascular lesions
- Dermatofibroma

The dataset is available for download from [ISIC Archive](https://api.isic-archive.com/collections/212/) or [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) (Take note that there is difference in metadata).

Note from dataset owner:
Attribution should be made by referencing the data descriptor manuscript: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018)

## Usage

1. **Data Understanding:**
   Run the `dataUnderstanding.py` script to explore and visualize the dataset:
   ```bash
   python dataUnderstanding.py
   ```

2. **Model Training:**
   Run the `modelTraining.py` script to preprocess the data, train the model, and evaluate its performance:
   ```bash
   python modelTraining.py
   ```

   After training, the model will be saved to a file for later use.

## Results

The project evaluates the model using accuracy, precision, recall, and other relevant metrics. The performance of the model on the test set can be visualized using the provided plotting functions.

## License

This project is licensed under the CC0 1.0 Universal License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The **HAM10000** dataset creators for providing a comprehensive dataset of dermatoscopic images.
- The TensorFlow and Keras communities for their excellent deep learning frameworks.
