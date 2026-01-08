import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import os
from typing import Optional
from .logger import logger

def plot_class_distribution(metadata: pd.DataFrame) -> None:
    """
    Plots the count of each lesion type.

    Args:
        metadata (pd.DataFrame): Metadata dataframe.
    """
    logger.info("Plotting class distribution...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='diagnosis', data=metadata)
    plt.title('Distribution of Lesion Types')
    plt.xlabel('Lesion Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

def plot_sample_images(metadata: pd.DataFrame, image_dir: str) -> None:
    """
    Plots a sample image from each class.

    Args:
        metadata (pd.DataFrame): Metadata dataframe.
        image_dir (str): Directory containing images.
    """
    logger.info("Plotting sample images...")
    classes = metadata['diagnosis'].unique()
    n_classes = len(classes)

    plt.figure(figsize=(15, 15))

    for i, c in enumerate(classes):
        try:
            image_path = metadata[metadata['diagnosis'] == c]['isic_id'].iloc[0]
            img_full_path = os.path.join(image_dir, image_path + '.jpg')
            
            if os.path.exists(img_full_path):
                img = Image.open(img_full_path)
                plt.subplot(1, n_classes, i + 1)
                plt.imshow(img)
                plt.title(c)
                plt.axis('off')
            else:
                logger.warning(f"Image not found at {img_full_path}")
        except Exception as e:
            logger.error(f"Error plotting sample for class {c}: {e}")

    plt.tight_layout()
    plt.show()

def plot_image_size_distribution(metadata: pd.DataFrame, image_dir: str) -> None:
    """
    Plots the distribution of image widths, heights, and aspect ratios.

    Args:
        metadata (pd.DataFrame): Metadata dataframe.
        image_dir (str): Image directory.
    """
    logger.info("Analyzing image size distribution...")
    image_sizes = []
    aspect_ratios = []
    
    # Process only first 1000 to be faster if dataset is large, or all if feasible.
    # Sticking to all for now but careful of perf.
    
    count = 0 
    for isic_id in metadata['isic_id']:
        img_path = os.path.join(image_dir, isic_id + '.jpg')
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                width, height = img.size
                image_sizes.append((width, height))
                aspect_ratios.append(width / height)
                count += 1
        
        # Optional: Break if too many for quick analysis
        # if count > 1000: break 

    if not image_sizes:
        logger.warning("No images found to analyze sizes.")
        return

    image_sizes_df = pd.DataFrame(image_sizes, columns=['Width', 'Height'])
    aspect_ratios_df = pd.Series(aspect_ratios, name='Aspect Ratio')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(image_sizes_df['Width'], kde=True, color='blue', label='Width')
    sns.histplot(image_sizes_df['Height'], kde=True, color='red', label='Height')
    plt.title('Distribution of Image Width and Height')
    plt.xlabel('Pixels')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(aspect_ratios_df, kde=True)
    plt.title('Distribution of Aspect Ratios')
    plt.xlabel('Aspect Ratio')

    plt.tight_layout()
    plt.show()

def plot_age_distribution(metadata: pd.DataFrame) -> None:
    """
    Plots the distribution of patient ages.

    Args:
        metadata (pd.DataFrame): Metadata dataframe.
    """
    logger.info("Plotting age distribution...")
    plt.figure(figsize=(10, 5))
    sns.histplot(metadata['age_approx'], bins=30, kde=True)
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

def plot_gender_distribution(metadata: pd.DataFrame) -> None:
    """
    Plots the distribution of gender.

    Args:
        metadata (pd.DataFrame): Metadata dataframe.
    """
    logger.info("Plotting gender distribution...")
    plt.figure(figsize=(5, 5))
    sns.countplot(x='sex', data=metadata)
    plt.title('Distribution of Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

def plot_correlation_matrix(metadata: pd.DataFrame) -> None:
    """
    Plots the correlation matrix of numeric features.

    Args:
        metadata (pd.DataFrame): Metadata dataframe.
    """
    logger.info("Plotting correlation matrix...")
    meta_copy = metadata.copy()
    if 'age_approx' in meta_copy.columns:
        meta_copy['age_approx'] = meta_copy['age_approx'].fillna(meta_copy['age_approx'].median())
    
    correlation_matrix = meta_copy.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

def plot_pairplot(metadata: pd.DataFrame) -> None:
    """
    Plots pairplot to explore relationships.

    Args:
        metadata (pd.DataFrame): Metadata dataframe.
    """
    logger.info("Plotting pairplot...")
    sns.pairplot(metadata, hue='diagnosis', vars=['age_approx'], palette='husl')
    plt.show()

def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Plots accuracy and loss from training history.

    Args:
        history (tf.keras.callbacks.History): Training history object.
    """
    logger.info("Plotting training history...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()
