import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Load emotion labels
with open("data/emotions.txt") as f:
    emotions = [line.strip() for line in f]
num_labels = len(emotions)

# Load thresholds
with open("src/optimal_thresholds.json") as f:
    thresholds = json.load(f)
threshold_array = np.array([thresholds.get(label, 0.5) for label in emotions])

# Load test data
df = pd.read_csv("data/test.tsv", sep="\t", names=["text", "labels", "id"])
df["labels"] = df["labels"].apply(lambda x: [int(i) for i in x.split(",")])
test_ds = Dataset.from_pandas(df)

# Load model & tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("goemotions-bert-v2")
model.eval()

# Encode test texts
inputs = tokenizer(list(test_ds["text"]), truncation=True, padding=True, return_tensors="pt")

# Get ground truth
y_true = torch.zeros((len(test_ds), num_labels))
for i, label_ids in enumerate(test_ds["labels"]):
    y_true[i][label_ids] = 1.0
y_true = y_true.numpy()

# Get predictions
with torch.no_grad():
    probs = torch.sigmoid(model(**inputs).logits).numpy()

# Apply thresholds
y_pred = (probs > threshold_array).astype(int)

# Evaluation
print("📊 Evaluation Metrics on Test Set:")
print("Micro F1:", f1_score(y_true, y_pred, average="micro"))
print("Micro Precision:", precision_score(y_true, y_pred, average="micro"))
print("Micro Recall:", recall_score(y_true, y_pred, average="micro"))

print("\n📋 Per-label report:")
print(classification_report(y_true, y_pred, target_names=emotions))
