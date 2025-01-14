import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import kornia as K
from kornia.feature import LoFTR
from typing import List
from kornia_moons.viz import draw_LAF_matches

class ImageMatcher:
    def __init__(self, model: LoFTR, max_images: int = None):
        """
        Initializes the ImageMatcher class.

        Args:
            model: Pretrained LoFTR model for matching.
            max_images: Maximum number of images to process from the directory. If None, all images are processed.
        """
        self.model = model # The pretrained LoFTR model for feature matching
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the device to GPU if available
        self.model = self.model.to(self.device)
        self.max_images = max_images # Max number of images to process, None means process all

    # Converts an image (NumPy array) to a PyTorch tensor.
    @staticmethod
    def image_to_tensor(img: np.ndarray) -> torch.Tensor:
        image = K.utils.image_to_tensor(img) # Convert NumPy array to tensor
        image = image.float().unsqueeze(dim=0) / 255.0 # Normalize and add batch dimension
        return image

    # Processes two images and returns the keypoint correspondences.
    def process_images(self, img1: np.ndarray, img2: np.ndarray):
        # Convert images to tensors and move them to the device
        img1_tensor = self.image_to_tensor(img1).to(self.device)
        img2_tensor = self.image_to_tensor(img2).to(self.device)
        
        # Prepare the input for the model (convert to grayscale)
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1_tensor),
            "image1": K.color.rgb_to_grayscale(img2_tensor),
        }
        
        with torch.no_grad():
            correspondences = self.model({k: v.to(self.device) for k, v in input_dict.items()})
        
        # Extract keypoints from the correspondences
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        
        return mkpts0, mkpts1, img1_tensor, img2_tensor

    # Finds the fundamental matrix and returns inliers
    def find_fundamental_matrix(self, mkpts0: np.ndarray, mkpts1: np.ndarray):
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        return Fm, inliers > 0 # Return inliers where the fundamental matrix is valid

    # Visualizes the matching keypoints between two images and saves the result.
    def visualize_matches(self, mkpts0: np.ndarray, mkpts1: np.ndarray, inliers: np.ndarray, img1_tensor: torch.Tensor, img2_tensor: torch.Tensor, iteration: int):
        # Drawing the matches (using the same code as before)
        #print(f"Image 1 shape (after conversion): {K.tensor_to_image(img1_tensor).shape}")
        #print(f"Image 2 shape (after conversion): {K.tensor_to_image(img2_tensor).shape}")

        #print(f"mkpts0 shape: {mkpts0.shape}")
        #print(f"mkpts1 shape: {mkpts1.shape}")
        #print(f"Inliers: {inliers}")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)

        # Draw the matching keypoints using the LAF (Local Affine Frames) visualization
        draw_LAF_matches(
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(mkpts0).view(1, -1, 2),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1),
            ),
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(mkpts1).view(1, -1, 2),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1),
            ),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(img1_tensor),
            K.tensor_to_image(img2_tensor),
            inliers,
            draw_dict={
                "inlier_color": (0.1, 1, 0.1, 0.5),
                "tentative_color": None,
                "feature_color": (0.2, 0.2, 1, 0.5),
                "vertical": False,
            },
            ax = ax,
        )
        
        # Save the matching result as an image
        filename = f"infer_results\matching_result_{iteration}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Image saved as '{filename}'.")

    # Full image processing pipeline: processes images, finds correspondences, and visualizes matches.
    def process_and_visualize_matching(self, img1: np.ndarray, img2: np.ndarray, iteration: int) -> None:
        mkpts0, mkpts1, img1_tensor, img2_tensor = self.process_images(img1, img2)
        Fm, inliers = self.find_fundamental_matrix(mkpts0, mkpts1)
        self.visualize_matches(mkpts0, mkpts1, inliers, img1_tensor, img2_tensor, iteration)

    # Processes all image pairs from a given directory and visualizes matches.
    def process_images_from_directory(self, directory: str) -> None:
        # Get all image files with .npy extension in the specified directory
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        
        # Limit the number of images if max_images is set
        if self.max_images is not None:
            image_files = image_files[:self.max_images]
        
        image_tensors = [np.load(img_file) for img_file in image_files]

        # Process image pairs and visualize matching
        for i in range(0, len(image_tensors) - 1, 2):
            img1 = image_tensors[i]
            img2 = image_tensors[i + 1]
            self.process_and_visualize_matching(img1, img2, i)