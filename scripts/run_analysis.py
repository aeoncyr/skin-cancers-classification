import sys
import os

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from skin_cancer_detection import data, visualization

def main():
    try:
        print("Loading metadata...")
        metadata = data.load_metadata()
        
        print("Plotting class distribution...")
        visualization.plot_class_distribution(metadata)
        
        print("Plotting sample images...")
        visualization.plot_sample_images(metadata, data.config.IMAGE_DIR)
        
        print("Plotting image size distribution...")
        visualization.plot_image_size_distribution(metadata, data.config.IMAGE_DIR)
        
        print("Plotting age distribution...")
        visualization.plot_age_distribution(metadata)
        
        print("Plotting gender distribution...")
        visualization.plot_gender_distribution(metadata)
        
        print("Plotting correlation matrix...")
        visualization.plot_correlation_matrix(metadata)
        
        print("Plotting pairplot...")
        visualization.plot_pairplot(metadata)
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
