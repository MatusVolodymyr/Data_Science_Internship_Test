import torch
from transformers import DistilBertTokenizerFast
from datasets import load_dataset

# Load a named entity recognition dataset from the Hugging Face Datasets library
dataset = load_dataset("telord/mountains-ner-dataset")

# Load a pre-trained DistilBERT tokenizer for tokenizing the input text
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

# Function to tokenize input text and align labels for token classification tasks
def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply the tokenization and label alignment function to the dataset in a batched manner
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Save the processed tokenized dataset to disk for later use
tokenized_datasets.save_to_disk('./tokenized_dataset')
