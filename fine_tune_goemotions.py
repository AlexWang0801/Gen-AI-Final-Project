import torch
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ✅ Use Apple MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ✅ Load emotion labels
with open("data/emotions.txt") as f:
    emotions = [line.strip() for line in f]
num_labels = len(emotions)

# ✅ Load TSV files
def load_data(split):
    df = pd.read_csv(f"data/{split}.tsv", sep="\t", names=["text", "labels", "id"])
    df["labels"] = df["labels"].apply(lambda x: [int(i) for i in x.split(",")])
    return Dataset.from_pandas(df)

train_ds = load_data("train")
dev_ds = load_data("dev")

# ✅ Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_and_encode(batch):
    encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    labels = torch.zeros((len(batch["labels"]), num_labels), dtype=torch.float32)  # 👈 ensure float32
    for i, label_ids in enumerate(batch["labels"]):
        labels[i][label_ids] = 1.0
    encoding["labels"] = labels
    return encoding


train_ds = train_ds.map(tokenize_and_encode, batched=True, remove_columns=["text", "id", "labels"])
dev_ds = dev_ds.map(tokenize_and_encode, batched=True, remove_columns=["text", "id", "labels"])

# ✅ Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")
model.to(device)

# ✅ Metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
    labels = torch.tensor(labels)
    return {
        "f1": f1_score(labels, preds, average="micro"),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
    }

from transformers import default_data_collator

def custom_collator(features):
    batch = default_data_collator(features)
    batch["labels"] = batch["labels"].float()  # ✅ Force labels to be float32
    return batch


# ✅ Training setup
args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics,
    data_collator=custom_collator
)

# ✅ Train
trainer.train()
trainer.save_model("goemotions-bert-v1")
print("✅ Model saved to goemotions-v1")
