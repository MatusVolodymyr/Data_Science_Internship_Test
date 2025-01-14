import torch
from transformers import DistilBertTokenizerFast
from datasets import load_dataset

# Load dataset
dataset = load_dataset("telord/mountains-ner-dataset")

# Load the fast tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

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

# Apply tokenization and alignment to the dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Save the tokenized dataset
tokenized_datasets.save_to_disk('./tokenized_dataset')
