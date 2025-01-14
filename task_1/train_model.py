import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import load_from_disk
import evaluate

# Define the list of labels for token classification
label_list = ["O", "B-mountain", "I-mountain"]

# Load the pre-processed tokenized dataset from disk
tokenized_datasets =  load_from_disk('./tokenized_dataset')

# Load the seqeval metric for evaluating token classification tasks
seq_eval = evaluate.load("seqeval")

# Load the pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

# Load the pre-trained DistilBERT model for token classification with a specified number of labels
model = DistilBertForTokenClassification.from_pretrained(
    'distilbert-base-cased',
    num_labels=len(label_list) 
)

# Define an early stopping callback to stop training if no improvement is seen after a certain number of epochs
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# Create a data collator that dynamically pads inputs during batching
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',             # Directory to save model checkpoints and results
    eval_strategy='epoch',              # Evaluate the model at the end of each epoch
    save_strategy="epoch",              # Save the model at the end of each epoch
    learning_rate=2e-5,                 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=15,
    weight_decay=0.01,
    save_total_limit=1,                 # Limit the total number of saved checkpoints
    logging_dir='./logs',
    logging_steps=10,                   # Log every 10 steps
    load_best_model_at_end=True,        # Load the best model at the end of training
    metric_for_best_model="accuracy",   # Metric used to evaluate the best model
    greater_is_better=True              # Higher metric value indicates better performance
)

# Define a function to compute evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)
    
    # Remove ignored index (-100) and convert to label names
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Compute the evaluation metrics using seqeval
    results = seq_eval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# Create a Trainer instance to manage training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator, 
    callbacks=[early_stopping],          # Add early stopping callback
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
test_results = trainer.evaluate(tokenized_datasets['test'])

# Print the results
print("Test set evaluation results:")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")

# Save the trained model and tokenizer to disk
model.save_pretrained('./ner_model')
tokenizer.save_pretrained('./ner_model')
