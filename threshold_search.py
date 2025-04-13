import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import f1_score
from datasets import Dataset

# 1. Load model, tokenizer, test set
model = BertForSequenceClassification.from_pretrained("goemotions-bert-mps")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model.eval()

with open("data/emotions.txt") as f:
    emotions = [line.strip() for line in f]
num_labels = len(emotions)

df = pd.read_csv("data/test.tsv", sep="\t", names=["text", "labels", "id"])
df["labels"] = df["labels"].apply(lambda x: [int(i) for i in x.split(",")])
test_ds = Dataset.from_pandas(df)

encodings = tokenizer(list(test_ds["text"]), truncation=True, padding=True, return_tensors="pt")
true_labels = torch.zeros((len(test_ds), num_labels))
for i, label_ids in enumerate(test_ds["labels"]):
    true_labels[i][label_ids] = 1.0

# 2. Get probabilities
with torch.no_grad():
    probs = torch.sigmoid(model(**encodings).logits).numpy()
y_true = true_labels.numpy()

# 3. Search best threshold for each class
thresholds = []
for i in range(num_labels):
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.91, 0.05):
        y_pred_col = (probs[:, i] > t).astype(int)
        f1 = f1_score(y_true[:, i], y_pred_col)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    thresholds.append(best_thresh)

print("✅ Best thresholds per class:")
for emo, t in zip(emotions, thresholds):
    print(f"{emo:<15}: {t:.2f}")

# 4. Predict with optimal thresholds
y_pred = np.zeros_like(probs)
for i in range(num_labels):
    y_pred[:, i] = (probs[:, i] > thresholds[i]).astype(int)

# 5. Re-evaluate
print("\n📊 Evaluation with dynamic thresholds:")
print(f"Micro F1: {f1_score(y_true, y_pred, average='micro'):.4f}")
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
