# Task 1: Named Entity Recognition (NER) for Mountain Names

This task involves training a Named Entity Recognition (NER) model to identify mountain names within text. The solution includes dataset preparation, model training, and inference demonstration.

## Project Structure

```
.
├── dataset_creation.py        # Script for dataset loading and tokenization
├── train_model.py             # Script for training the NER model
├── infer_model.py             # Script for model inference
├── tokenized_dataset/         # Directory containing the tokenized dataset
├── dataset/                   # Directory containing the initial dataset
├── ner_model/                 # Directory for the trained model and tokenizer (will be automatically created after running `train_model.py`)
├── requirements.txt           # List of dependencies
├── requirements.txt           # List of dependencies(for conda)
├── intern-task-1_demo.ipynb   # Kaggle jupyter notebook with demo 
└── README.md                  # This README file
```

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

### Files and Directories

- `dataset_creation.py`: This script loads the dataset and applies tokenization, saving the processed data for model training. (bataset is automaticly loaded from huggingface)
- `train_model.py`: This script defines the model architecture, training parameters, and performs model training.
- `infer_model.py`: This script loads the trained model and performs inference on new text inputs.
- `tokenized_dataset/`: Directory where the tokenized dataset is saved.
- `ner_model/`: Directory where the trained model and tokenizer are saved (It will be automatically created after running `train_model.py`, or you can manually create it by downloading the weights from [this Google Drive link](https://drive.google.com/drive/folders/1m9i5t5lUgy_DTjf05jiRSyQaDfLV_lAS) to skip the training process).
- `intern_task_1_demo.ipynb`: Jupyter notebook demonstrating the end-to-end process.

## Usage

### 1. Dataset Creation (Optional): If you need to recreate the dataset, run:

Run the `dataset_creation.py` script to load and preprocess the dataset:

```bash
python dataset_creation.py
```

### 2. Model Training

Run the `train_model.py` script to train the model:

```bash
python train_model.py
```

This will save the trained model and tokenizer in the `ner_model/` directory.

### 3. Model Inference

Run the `infer_model.py` script to perform inference on new text inputs:

```bash
python infer_model.py
```

You can modify the text input in the script to test different sentences.



