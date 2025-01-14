from kornia.feature import LoFTR
from model import ImageMatcher

# Initialize LoFTR with a pre-trained model
loftr = LoFTR(pretrained='outdoor')

# Create an instance of ImageMatcher with a maximum of (for now only 2, cause i dont  have cuda installed....) images to process
image_matcher = ImageMatcher(model=loftr, max_images=2)

# Directory containing saved images as .npy files
image_directory = './processed_images'

# Process and visualize matches for images in the directory
image_matcher.process_images_from_directory(image_directory)
