import os
import numpy as np
import rasterio
import cv2
from typing import List
from rasterio.plot import reshape_as_image

# Class to load and process Sentinel-2 images
class Sentinel2ImageLoader:
    def __init__(self, base_dir: str, max_dimension: int = 1024, save_dir: str = './processed_images'):
        """
        Initializes the Sentinel2ImageLoader class.

        Args:
            base_dir: Directory where the image files are located.
            max_dimension: Maximum dimension (height/width) for resizing.
            save_dir: Directory to save the processed images.
        """
        self.base_dir = base_dir
        self.max_dimension = max_dimension
        self.save_dir = save_dir
        self.image_files = self._find_image_files()     # List of image file paths

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    # Method to find all image files in the base directory
    def _find_image_files(self) -> List[str]:
        return [
            os.path.join(self.base_dir, file)
            for file in os.listdir(self.base_dir)
            if file.endswith('TCI.jp2')
        ]

    # Method to load an image using rasterio
    def _load_image_rasterio(self, image_path: str) -> np.ndarray:
        with rasterio.open(image_path, "r", driver='JP2OpenJPEG') as src:
            raster_image = src.read()  # Shape: (bands, height, width)
        return self._resize_image(raster_image)

    # Method to resize the image to a specified maximum dimension
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        img = reshape_as_image(img)     # Convert the image to shape (height, width, bands)
        height, width = img.shape[:2]   # Extract height and width of the image
        scale = self.max_dimension / max(height, width) # Calculate scaling factor to fit max dimension
        new_height, new_width = int(height * scale), int(width * scale)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img

    # Method to load all images from the directory, resize them, and save as NumPy arrays
    def load_all_images(self):
        for image_file in self.image_files:
            img = self._load_image_rasterio(image_file)
            # Save the processed image as a NumPy array (.npy file)
            file_name = os.path.splitext(os.path.basename(image_file))[0]
            np.save(os.path.join(self.save_dir, f"{file_name}.npy"), img)
            print(f"Saved {file_name}.npy")


loader = Sentinel2ImageLoader("dataset")
loader.load_all_images()
