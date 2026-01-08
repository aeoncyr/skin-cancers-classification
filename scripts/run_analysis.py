import sys
import os

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from skin_cancer_detection import data, visualization, config
from skin_cancer_detection.logger import logger

def main():
    try:
        logger.info("Starting analysis...")
        logger.info("Loading metadata...")
        metadata = data.load_metadata()
        
        logger.info("Plotting class distribution...")
        visualization.plot_class_distribution(metadata)
        
        logger.info("Plotting sample images...")
        visualization.plot_sample_images(metadata, config.IMAGE_DIR)
        
        logger.info("Plotting image size distribution...")
        visualization.plot_image_size_distribution(metadata, config.IMAGE_DIR)
        
        logger.info("Plotting age distribution...")
        visualization.plot_age_distribution(metadata)
        
        logger.info("Plotting gender distribution...")
        visualization.plot_gender_distribution(metadata)
        
        logger.info("Plotting correlation matrix...")
        visualization.plot_correlation_matrix(metadata)
        
        logger.info("Plotting pairplot...")
        visualization.plot_pairplot(metadata)
        
        logger.info("Analysis complete.")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}", exc_info=True)

if __name__ == "__main__":
    main()
