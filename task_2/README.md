# Project Title: Data Science Internship Task 2

## Overview
This project is a part of the Data Science Internship Task 2, designed to process satellite images, create datasets, and infer models for analysis. The main components include dataset creation, model inference, and demonstration of results through visualizations.

## Project Structure
```
H:\DATA_SCIENCE_INTERNSHIP_TEST\TASK_2
├── dataset_creation.py         # Script for dataset loading and preprocessing
├── infer_model.py              # Script for model inference
├── intern_task_2_demo.ipynb    # Kaggle jupyter notebook with demo
├── model.py                    # Script for dataset loading and preprocessing
├── README.md                   # This README file
├── requirements.txt            # List of dependencies
├── requirements_conda.txt      # List of dependencies(for conda)
├── dataset/ (in .gitignore)    # Directory for the initial dataset
├── processed_images/           # Directory containing preprocessed dataset
|       T36UXA_20180726T084009_TCI.npy
|       ...
|
├── infer_results/              # Directory containing the match results 
|       matching_result_0.png


## Setup Instructions

### Requirements

Ensure you have Python 3 installed. Install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
Alternatively, you can create a Conda environment:
```bash
conda create --name env_name --file requirements_conda.txt
```

## Files and Directories
- `dataset_creation.py`: This script preprocesses satellite images from datasets. Calling this script is optional as the processed images are already available in the `processed_images` folder.
- `infer_model.py`: This script runs model inference on the preprocessed dataset.
- `intern_task_2_demo.ipynb`: Jupyter notebook demonstrating the end-to-end process.
- `model.py`: Contains the model definition and related utilities.
- `README.md`: Project documentation.
- `dataset`: Contains raw satellite RGB images (excluded from version control due to large size, can be manually downloaded from [this Google Drive link](https://drive.google.com/file/d/19h0kRre-dTeIt7AgvpBomxlqChK6uQEw/view?usp=sharing) to run `dataset_creation.py`).
- `infer_results`: Stores preprocessed images in `.npy` format.
- `results`: Stores output images, including match results.

## Usage
### 1. Dataset Creation (Optional): If you need to recreate the dataset, run:
Run the `dataset_creation.py` script to generate datasets from the provided images. Ensure that the `dataset` directory is populated with the necessary `.jp2` files.
```bash
python dataset_creation.py
```
Ensure the dataset/ directory contains the necessary .jp2 files.

### Model Inference
Use the `infer_model.py` script to perform inference using the pre-trained model defined in `model.py`.
```bash
python infer_model.py
```
The results will be saved in the `infer_results/` directory.

## Notes
- The `dataset` directory is excluded from version control due to the large size of the images. Users must manually populate this directory with the necessary image files from [this Google Drive link](https://drive.google.com/file/d/19h0kRre-dTeIt7AgvpBomxlqChK6uQEw/view?usp=sharing) before running the `dataset_creation.py` script.



