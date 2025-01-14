# Task 1: Named Entity Recognition (NER) for Mountain Names

This task involves training a Named Entity Recognition (NER) model to identify mountain names within text. The solution includes dataset preparation, model training, and inference demonstration.

## Project Structure

```
.
├── dataset_creation.py        # Script for dataset loading and tokenization
├── train_model.py             # Script for training the NER model
├── infer_model.py             # Script for model inference
├── tokenized_dataset/         # Directory containing the tokenized dataset
├── ner_model/                 # Directory for the trained model and tokenizer
├── requirements.txt           # List of dependencies
├── requirements.txt           # List of dependencies(for conda usage)
└── README.md                  # This README file
```

## Setup Instructions

### Requirements

Ensure you have Python 3 installed. Install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
or if youare using conda:
```bash
conda create --name <env> --file requirements_conda.txt
```

### Files and Directories

- `dataset_creation.py`: This script loads the dataset and applies tokenization, saving the processed data for model training.
- `train_model.py`: This script defines the model architecture, training parameters, and performs model training.
- `infer_model.py`: This script loads the trained model and performs inference on new text inputs.
- `tokenized_dataset/`: Directory where the tokenized dataset is saved.
- `ner_model/`: Directory where the trained model and tokenizer are saved(will be created automaticly after train_model.py is executed, or can be created to load weights manualy from https://drive.google.com/drive/folders/1m9i5t5lUgy_DTjf05jiRSyQaDfLV_lAS, to skipp the training process).

## Usage

### 1. Dataset Creation

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

## Improvements

For future enhancements, consider the following:

1. **Dataset Expansion**: Increase the size and diversity of the dataset to improve model generalization.
2. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and number of epochs to optimize performance.
3. **Data Augmentation**: Use techniques like synonym replacement or back-translation to augment the training data.

## Conclusion

This project demonstrates the workflow for developing an NER model using Python and transformer-based architectures. The provided scripts ensure a modular and clear approach to dataset preparation, model training, and inference.

