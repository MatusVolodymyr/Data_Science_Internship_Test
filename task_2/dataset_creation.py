import os
import numpy as np
import rasterio
import cv2
from typing import List
from rasterio.plot import reshape_as_image

class Sentinel2ImageLoader:
    def __init__(self, base_dir: str, max_dimension: int = 1024, save_dir: str = './processed_images'):
        self.base_dir = base_dir
        self.max_dimension = max_dimension
        self.save_dir = save_dir
        self.image_files = self._find_image_files()

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _find_image_files(self) -> List[str]:
        return [
            os.path.join(self.base_dir, file)
            for file in os.listdir(self.base_dir)
            if file.endswith('TCI.jp2')
        ]

    def _load_image_rasterio(self, image_path: str) -> np.ndarray:
        with rasterio.open(image_path, "r", driver='JP2OpenJPEG') as src:
            raster_image = src.read()  # Shape: (bands, height, width)
        return self._resize_image(raster_image)

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        img = reshape_as_image(img)
        height, width = img.shape[:2]
        scale = self.max_dimension / max(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img

    def load_all_images(self):
        for image_file in self.image_files:
            img = self._load_image_rasterio(image_file)
            # Save the image as a NumPy array
            file_name = os.path.splitext(os.path.basename(image_file))[0]
            np.save(os.path.join(self.save_dir, f"{file_name}.npy"), img)
            print(f"Saved {file_name}.npy")

# Usage example
loader = Sentinel2ImageLoader("dataset")
loader.load_all_images()
