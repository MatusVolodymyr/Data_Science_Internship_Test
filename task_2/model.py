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
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.max_images = max_images

    @staticmethod
    def image_to_tensor(img: np.ndarray) -> torch.Tensor:
        """Converts an image to a tensor."""
        image = K.utils.image_to_tensor(img)
        image = image.float().unsqueeze(dim=0) / 255.0
        return image

    def process_images(self, img1: np.ndarray, img2: np.ndarray):
        """Processes two images and returns the correspondences."""
        img1_tensor = self.image_to_tensor(img1).to(self.device)
        img2_tensor = self.image_to_tensor(img2).to(self.device)
        
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1_tensor),
            "image1": K.color.rgb_to_grayscale(img2_tensor),
        }
        
        with torch.no_grad():
            correspondences = self.model({k: v.to(self.device) for k, v in input_dict.items()})
        
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        
        return mkpts0, mkpts1, img1_tensor, img2_tensor

    def find_fundamental_matrix(self, mkpts0: np.ndarray, mkpts1: np.ndarray):
        """Finds the fundamental matrix and returns inliers."""
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        return Fm, inliers > 0

    def visualize_matches(self, mkpts0: np.ndarray, mkpts1: np.ndarray, inliers: np.ndarray, img1_tensor: torch.Tensor, img2_tensor: torch.Tensor, iteration: int):
        """Visualizes the matching keypoints between two images."""
        # Drawing the matches (using the same code as before)
        #print(f"Image 1 shape (after conversion): {K.tensor_to_image(img1_tensor).shape}")
        #print(f"Image 2 shape (after conversion): {K.tensor_to_image(img2_tensor).shape}")

        #print(f"mkpts0 shape: {mkpts0.shape}")
        #print(f"mkpts1 shape: {mkpts1.shape}")
        #print(f"Inliers: {inliers}")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)

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
        
        # Save the figure
        filename = f"infer_results\matching_result_{iteration}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Image saved as '{filename}'.")


    def process_and_visualize_matching(self, img1: np.ndarray, img2: np.ndarray, iteration: int) -> None:
        """Full image processing pipeline: processes images, finds correspondences, and visualizes matches."""
        mkpts0, mkpts1, img1_tensor, img2_tensor = self.process_images(img1, img2)
        Fm, inliers = self.find_fundamental_matrix(mkpts0, mkpts1)
        self.visualize_matches(mkpts0, mkpts1, inliers, img1_tensor, img2_tensor, iteration)

    def process_images_from_directory(self, directory: str) -> None:
        """Processes all image pairs from a given directory and visualizes matches."""
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        
        # Limit the number of images if max_images is set
        if self.max_images is not None:
            image_files = image_files[:self.max_images]
        
        image_tensors = [np.load(img_file) for img_file in image_files]

        for i in range(0, len(image_tensors) - 1, 2):
            img1 = image_tensors[i]
            img2 = image_tensors[i + 1]
            self.process_and_visualize_matching(img1, img2, i)