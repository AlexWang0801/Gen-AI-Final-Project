# fine_tune_goemotions_v2.py
import torch
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch import nn
from torch.nn import functional as F
from transformers import default_data_collator

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
    labels = torch.zeros((len(batch["labels"]), num_labels), dtype=torch.float32)
    for i, label_ids in enumerate(batch["labels"]):
        labels[i][label_ids] = 1.0
    encoding["labels"] = labels.numpy().tolist()  # convert to plain list for Hugging Face
    return encoding

train_ds = train_ds.map(tokenize_and_encode, batched=True, remove_columns=["text", "id", "labels"])
dev_ds = dev_ds.map(tokenize_and_encode, batched=True, remove_columns=["text", "id", "labels"])

# ✅ Custom loss: Focal Loss for multi-label classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.to(inputs.dtype)  # make sure type matches
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# ✅ Model
def get_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model

model = get_model()
loss_fn = FocalLoss()

# ✅ Custom Trainer with Focal Loss
class FocalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = torch.tensor(inputs.pop("labels"), dtype=torch.float32)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

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

# ✅ Training setup
args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    logging_dir="logs",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = FocalTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator
)

# ✅ Train and save
trainer.train()
trainer.save_model("goemotions-bert-v2")
print("✅ Model saved to goemotions-bert-v2")
