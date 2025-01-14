import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
from datasets import load_from_disk

tokenized_datasets =  load_from_disk('./tokenized_dataset')

label_list = ["O", "B-mountain", "I-mountain"]

# Creating label2id and id2label mappings
id2label = {idx: label for idx, label in enumerate(label_list)}

# Loading the trained model and tokenizer
model = DistilBertForTokenClassification.from_pretrained('./ner_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./ner_model')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def predict(text):
    tokens = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}
    output = model(**tokens)
    logits = output.logits
    predictions = torch.argmax(logits, dim=2)
    return predictions

def format_inference_results(text, predictions, tokenizer, id2label):
    # Tokenize input text to get the tokenization alignment
    tokens = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    tokens_decoded = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    
    # Convert predictions from tensor to list
    predictions = predictions[0].cpu().numpy()
    
    # Map the predictions to the corresponding labels
    predicted_labels = [id2label[pred] for pred in predictions]
    
    # Combine tokens with their corresponding labels
    result = []
    for token, label in zip(tokens_decoded, predicted_labels):
        # Ignore special tokens like [CLS] and [SEP]
        if token not in tokenizer.all_special_tokens:
            result.append((token, label))
    
    return result

def generate_output(text, tokenizer, id2label):
    tokens = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}  # Ensure tokens are on the correct device
    output = model(**tokens)
    logits = output.logits
    predictions = torch.argmax(logits, dim=2)
    
    # Call the function to format results
    formatted_results = format_inference_results(text, predictions, tokenizer, id2label)
    
    # Print the formatted results
    for token, label in formatted_results:
        print(f"{token}: {label}")

# Example usage
text = "Tasman Ridge is a ridge , 3 nautical miles ( 6 km ) long , located 10 nautical miles ( 18 km ) northeast of Mount Hooker , bounded on the northwest by Ball Glacier and on the southeast by Hooker Glacier , descending into Blue Glacier in the Royal Society Range , Victoria Land."
generate_output(text, tokenizer, id2label)
