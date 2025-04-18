# fine_tune_goemotions_clean.py
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ✅ Load emotion labels
with open("data/emotions.txt") as f:
    emotions = [line.strip() for line in f]
num_labels = len(emotions)

# ✅ Load dataset
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
    labels = torch.zeros((len(batch["labels"]), num_labels), dtype=torch.float32)
    for i, label_ids in enumerate(batch["labels"]):
        labels[i][label_ids] = 1.0
    
    encoding["labels"] = [list(map(float, row)) for row in labels.tolist()]
    return encoding


train_ds = train_ds.map(tokenize_and_encode, batched=True, remove_columns=["text", "id", "labels"])
dev_ds = dev_ds.map(tokenize_and_encode, batched=True, remove_columns=["text", "id", "labels"])

# ✅ Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# ✅ Metrics
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int()
    labels = torch.tensor(labels).float()  # ensure float type
    return {
        "f1": f1_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
        "accuracy": accuracy_score(labels, preds),
    }

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    warmup_steps=300,
    logging_dir="logs",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# ✅ Use DataCollatorWithPadding to keep float label types
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ✅ Train & Save
trainer.train()
trainer.save_model("goemotions-bert-v3")
tokenizer.save_pretrained("goemotions-bert-v3")
print("✅ Model saved to goemotions-bert-v3")
