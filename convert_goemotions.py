import pandas as pd
import os

# Load all 3 raw GoEmotions CSV files
base_path = "data/"
csv_files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

df_list = [pd.read_csv(os.path.join(base_path, file)) for file in csv_files]
df = pd.concat(df_list).reset_index(drop=True)

# Drop examples that were marked as unclear
df = df[df["example_very_unclear"] == False]

# Load emotion labels and create label-to-index map
with open("data/emotions.txt") as f:
    emotions = [line.strip() for line in f]
emo2id = {e: i for i, e in enumerate(emotions)}

# Get examples where at least 2 raters agree on at least 1 emotion
df = df.groupby("id").filter(lambda x: len(x) >= 2)
grouped = df.groupby("id")

# Prepare rows for train/dev/test
rows = []
for example_id, group in grouped:
    row = group.iloc[0]  # any row works; text is the same
    labels = []
    for _, r in group.iterrows():
        labels += [emotions[i] for i, val in enumerate(r[10:]) if val == 1]
    # Get label counts, keep most common ones
    label_counts = pd.Series(labels).value_counts()
    final_labels = sorted(set(label_counts[label_counts > 1].index))
    if not final_labels:
        continue
    label_ids = [str(emo2id[label]) for label in final_labels]
    rows.append([row["text"], ",".join(label_ids), row["id"]])

# Shuffle and split
final_df = pd.DataFrame(rows, columns=["text", "emotion_ids", "comment_id"])
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_split = 0.8
dev_split = 0.1
n = len(final_df)
train_df = final_df.iloc[:int(train_split * n)]
dev_df = final_df.iloc[int(train_split * n):int((train_split + dev_split) * n)]
test_df = final_df.iloc[int((train_split + dev_split) * n):]

# Save the TSVs
train_df.to_csv("data/train.tsv", sep="\t", header=False, index=False)
dev_df.to_csv("data/dev.tsv", sep="\t", header=False, index=False)
test_df.to_csv("data/test.tsv", sep="\t", header=False, index=False)

print("✅ Done! Files saved to data/train.tsv, dev.tsv, and test.tsv")
