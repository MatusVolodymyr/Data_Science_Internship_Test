import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import load_from_disk
import evaluate

label_list = ["O", "B-mountain", "I-mountain"]

tokenized_datasets =  load_from_disk('./tokenized_dataset')

seqeval = evaluate.load("seqeval")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

# Loadimg the DistilBERT model for token classification
model = DistilBertForTokenClassification.from_pretrained(
    'distilbert-base-cased',
    num_labels=len(label_list) 
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator, 
    callbacks=[early_stopping],
)

# Train the model
trainer.train()

test_results = trainer.evaluate(tokenized_datasets['test'])

# Print the results
print("Test set evaluation results:")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")

# Save the model
model.save_pretrained('./ner_model')
tokenizer.save_pretrained('./ner_model')
