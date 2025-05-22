import os
import cv2
import numpy as np
from tqdm import tqdm

class DataLoader:
    def __init__(self, image_dir, label_dir, output_dir):
        """
        Data loader that converts images and labels into .npz format.
        
        Parameters:
        - image_dir: Directory containing image files
        - label_dir: Directory containing label files
        - output_dir: Directory to save the converted .npz files
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_as_npz(self, image, label, file_name):
        """
        Saves the image and label as a .npz file.
        
        Parameters:
        - image: Image data
        - label: Label data
        - file_name: Name of the file to save (without extension)
        """
        np.savez(os.path.join(self.output_dir, file_name), image=image, label=label)

    def convert_png_to_npz(self, img_path, label_path):
        """
        Converts a single image and its label to .npz format.
        
        Parameters:
        - img_path: Path to the image file
        - label_path: Path to the label file
        """
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

        label = cv2.imread(label_path, flags=0)  # Read the label in grayscale

        # Save the image and label as a .npz file
        file_name = os.path.basename(img_path).split('.')[0]  # Extract file name without extension
        self.save_as_npz(image, label, file_name)

    def m4d_label_mapping(self, label):
        """
        Label mapping for M4D mode.
        
        Parameters:
        - label: Raw label image
        
        Returns:
        - mapped label image
        """
        label = np.where((label >= 82) & (label <= 88), 0, label)   # Category 0 (82-88)
        label = np.where((label >= 91) & (label <= 172), 0, label)  # Category 0 (91-172)
        label = np.where(label <= 70, 0, label)                     # Category 0 (0-70)

        label = np.where((label >= 173) & (label <= 183), 1, label)  # Category 1 (173-183)
        label = np.where(label == 90, 3, label)                     # Category 3 (90)
        label = np.where(label == 89, 4, label)                     # Category 4 (89)
        label = np.where((label >= 71) & (label <= 81), 2, label)   # Category 2 (71-81)
        return label

    def sos_label_mapping(self, label):
        """
        Label mapping for SOS mode.
        
        Parameters:
        - label: Raw label image
        
        Returns:
        - mapped label image
        """
        label = np.where(label < 128, 0, 1)  # Category 0 if label < 128, else Category 1
        return label

    def process_data(self, mode='M4D'):
        """
        Processes all images and labels in the dataset, converts them to .npz format, and applies label mapping.
        
        Parameters:
        - mode: Mode for label mapping. Choose 'M4D' for M4D mode or 'SOS' for SOS mode.
        """
        files = os.listdir(self.image_dir)  # Get all image files

        # Loop through all image files and process them
        for img_path in tqdm(files, desc="Processing Data", leave=False):
            file_name = img_path.split('.')[0]  # Extract file name (without extension)
            img_path = os.path.join(self.image_dir, img_path)  # Full image file path
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.png')  # Full label file path

            # Process the image and label
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

            # Read the label
            label = cv2.imread(label_path, flags=0)

            # Apply the corresponding label mapping
            if mode == 'M4D':
                label = self.m4d_label_mapping(label)  # Apply M4D mode label mapping
            elif mode == 'SOS':
                label = self.sos_label_mapping(label)  # Apply SOS mode label mapping
            else:
                raise ValueError(f"Unknown mode: {mode}. Choose 'M4D' or 'SOS'.")

            # Save the image and label as .npz
            self.save_as_npz(image, label, file_name)

        print(f"Data processing completed. Files saved to {self.output_dir}")


def main():
    # Define the paths for the images, labels, and output directory
    root = os.path.dirname(__file__)
    image_dir = os.path.join(root, "aug_data", "images")
    label_dir = os.path.join(root, "aug_data", "labels")
    output_dir = os.path.join(root, "aug_data_npz")

    # Initialize the data loader
    data_loader = DataLoader(image_dir, label_dir, output_dir)

    # Process the data and save as .npz files (choose the mode: 'M4D' or 'SOS')
    data_loader.process_data(mode='M4D')  # Change to 'SOS' for SOS mode


if __name__ == "__main__":
    main()
