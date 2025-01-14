# Data Science Internship Test Project

This project is designed as part of a Data Science Internship test, focusing on two primary domains: Natural Language Processing (NLP) and Computer Vision (CV).

## Project Structure

```
.
├── task_1/                    # Folder containing Task 1 related files
│   ├── train_model.py         # Training script for the NER model
│   ├── infer_model.py         # Inference script for the NER model
│   ├── dataset_creation.py    # Script for dataset preparation
│   ├── requirements.txt       # Dependencies for Task 1
│   ├── README.md              # Detailed information about Task 1
│   └── ...                    # Other related files for Task 1
├── task_2/                    # Folder containing Task 2 related files
│   ├── infer_model.py         # Inference script for image matching model
│   ├── dataset_creation.py    # Script for dataset preparation
│   ├── requirements.txt       # Dependencies for Task 2
│   ├── README.md              # Detailed information about Task 2
│   └── ...                    # Other related files for Task 2
├── README.md                  # General information about the project
└── ...                        # Any additional files or folders
```

## General Information

This project is divided into two tasks:

### Task 1: Named Entity Recognition (NER)
This task involves training a Named Entity Recognition model to identify mountain names in text. It includes dataset preparation, model training, and inference. Specific details and instructions for Task 1 are provided in its respective `README.md` file.

### Task 2: Image Matching
This task involves developing an algorithm to match satellite images from different seasons using a pre-trained model. It includes image preprocessing, model training, and inference. Specific details and instructions for Task 2 are provided in its respective `README.md` file.

## Setup Instructions

### Virtual Environments
Each task has its own virtual environment to manage dependencies (projects were creted using conda envs):

1. **Task 1**:
   - Navigate to the `task_1/` directory.
   - Create and activate a virtual environment:
     ```bash
     python -m venv task1_env
     source task1_env/bin/activate
     ```
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```
    Alternatively, you can create a Conda environment:
    ```bash
    conda create --name task1_env --file requirements_conda.txt
    ```

2. **Task 2**:
   - Navigate to the `task_2/` directory.
   - Create and activate a virtual environment:
     ```bash
     python -m venv task2_env
     source task2_env/bin/activate
     ```
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```
    Alternatively, you can create a Conda environment:
    ```bash
    conda create --name task2_env --file requirements_conda.txt
    ```

### Running the Project
- Follow the specific instructions in each task's `README.md` file to run the scripts and understand the workflow for dataset preparation, model training, and inference.

## Notes
- Ensure you have Python 3 installed on your system.
- Each task has a `requirements.txt` file listing the required Python packages. Use these files to install dependencies in the respective virtual environments.
- Reports and suggestions for potential improvements are included within each task folder.


