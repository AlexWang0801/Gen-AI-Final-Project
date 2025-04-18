import torch
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset
import numpy as np

# --- Load labels ---
with open("data/emotions.txt") as f:
    emotions = [line.strip() for line in f]
num_labels = len(emotions)

# --- Load model and tokenizer ---
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("goemotions-bert-v2")
model.eval()

# --- Load test set ---
df = pd.read_csv("data/test.tsv", sep="\t", names=["text", "labels", "id"])
df["labels"] = df["labels"].apply(lambda x: [int(i) for i in x.split(",")])
test_ds = Dataset.from_pandas(df)

# --- Prepare inputs and true labels ---
encodings = tokenizer(list(test_ds["text"]), truncation=True, padding=True, return_tensors="pt")
true_labels = torch.zeros((len(test_ds), num_labels))
for i, label_ids in enumerate(test_ds["labels"]):
    true_labels[i][label_ids] = 1.0

# --- Get model predictions ---
with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()

# --- Convert to numpy for sklearn
y_true = true_labels.numpy()
y_pred = preds.numpy()

# --- Compute metrics
print("📊 Evaluation Metrics on Test Set（before add thresholds）:")
print(f"Micro F1: {f1_score(y_true, y_pred, average='micro'):.4f}")
print(f"Micro Precision: {precision_score(y_true, y_pred, average='micro'):.4f}")
print(f"Micro Recall: {recall_score(y_true, y_pred, average='micro'):.4f}")

# Optional: Per-label report
print("\n📋 Per-label report:")
print(classification_report(y_true, y_pred, target_names=emotions, zero_division=0))
