import numpy as np
import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 2. Define a function to compute metrics
def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    cm = confusion_matrix(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return acc, mcc, cm, f1


# Load and preprocess data
reuters = pd.read_pickle('reuters_news_concatenated.pkl', 'bz2').sample(frac=1).reset_index()
clusters = 4
labels = np.copy(reuters.Y)
labels[reuters.Y < np.percentile(reuters.Y, 100/clusters)] = 0
for i in range(1, clusters):
    labels[reuters.Y > np.percentile(reuters.Y, 100*i/clusters)] = i
reuters.Y = labels.astype("int")

# Create reuters_BERT dataframe
reuters_BERT = pd.DataFrame(columns=['label', 'text'])
reuters_BERT.label = reuters.Y
reuters_BERT.text = reuters.news.apply(lambda x: ' '.join(x))

# Tokenize and encode text data
tokenizer = AlbertTokenizer.from_pretrained('ALbert/spiece.model', do_lower_case=True)  # Changed to ALBERT tokenizer

# 1. Tokenize the texts
input_ids = [tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True) for text in reuters_BERT['text']]

# 2. Create attention masks
attention_masks = [[int(token_id > 0) for token_id in ids] for ids in input_ids]

# 3. Convert tokenized inputs into PyTorch tensors and pad them
input_ids = [torch.tensor(ids) for ids in input_ids]
input_ids_tensor = pad_sequence(input_ids, batch_first=True, padding_value=0)

# 4. Convert attention masks into PyTorch tensors and pad them
attention_masks = [torch.tensor(mask) for mask in attention_masks]
attention_masks_tensor = pad_sequence(attention_masks, batch_first=True, padding_value=0)

labels_tensor = torch.tensor(reuters_BERT['label'].values).long()
train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_ids_tensor, labels_tensor, test_size=0.3, random_state=42)
train_masks, temp_masks = train_test_split(attention_masks_tensor, test_size=0.3, random_state=42)
validation_inputs, test_inputs, validation_labels, test_labels = train_test_split(temp_inputs, temp_labels, test_size=2/3, random_state=42)
validation_masks, test_masks = train_test_split(temp_masks, test_size=2/3, random_state=42)

# DataLoader
batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Model and Optimizer
model_path = "ALbert"  # Changed to ALBERT model
model = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=clusters).to(device)  # Changed to ALBERT model

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training Loop
accs, mccs, cms, f1s = [], [], [], []
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    all_preds = []
    all_labels = []
    losses = []
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)


        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        labels = labels.to('cpu').numpy()
        logits = outputs[1].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels)
        losses.append(loss.item())
    # Compute metrics
    acc, mcc, cm, f1 = compute_metrics(all_preds, all_labels)
    output_path = "ALbertmodel_train_results.txt"
    with open(output_path, 'a') as f:
        f.write(f"train Accuracy: {acc:.4f}\n")
        f.write(f"train MCC: {mcc:.4f}\n")
        f.write(f"train F1 Score: {f1:.4f}\n")
        f.write(f"train losses: {np.mean(losses):.4f}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds))
        f.write("\nConfusion Matrix:\n")
        for row in cm:
            f.write(' '.join([str(x) for x in row]) + '\n')

    # Validation Loop
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0


    all_preds = []
    all_labels = []
    losses = []

    for batch in validation_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        total_eval_accuracy += np.sum(np.argmax(logits, axis=1) == labels)
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels)
        losses.append(loss.item())
    # Compute metrics
    acc, mcc, cm, f1 = compute_metrics(all_preds, all_labels)
    output_path = "ALbertmodel_evaluation_results.txt"
    with open(output_path, 'a') as f:
        f.write(f"Validation Accuracy: {acc:.4f}\n")
        f.write(f"Validation MCC: {mcc:.4f}\n")
        f.write(f"Validation F1 Score: {f1:.4f}\n")
        f.write(f"Validation losses: {np.mean(losses):.4f}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds))
        f.write("\nConfusion Matrix:\n")
        for row in cm:
            f.write(' '.join([str(x) for x in row]) + '\n')
    avg_val_accuracy = total_eval_accuracy / len(validation_inputs)
    print(f"Epoch: {epoch+1}, Validation Accuracy: {avg_val_accuracy:.4f}")
    print(classification_report(all_labels, all_preds))


# Save Model
losses = []
all_preds = []
all_labels = []
torch.save(model.state_dict(), "ALbert_model.pth")
# test
for batch in test_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    logits = outputs[1]
    total_eval_loss += loss.item()
    logits = logits.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()
    total_eval_accuracy += np.sum(np.argmax(logits, axis=1) == labels)
    preds = np.argmax(logits, axis=1)
    all_preds.extend(preds)
    all_labels.extend(labels)
    losses.append(loss.item())
# Compute metrics
acc, mcc, cm, f1 = compute_metrics(all_preds, all_labels)
output_path = "ALbertmodel_test_results.txt"
with open(output_path, 'w') as f:
    f.write(f"test Accuracy: {acc:.4f}\n")
    f.write(f"test MCC: {mcc:.4f}\n")
    f.write(f"test F1 Score: {f1:.4f}\n")
    f.write(f"test losses: {np.mean(losses):.4f}\n")
    f.write("test Classification Report:\n")
    f.write(classification_report(all_labels, all_preds))
    f.write("\nConfusion Matrix:\n")
    for row in cm:
        f.write(' '.join([str(x) for x in row]) + '\n')