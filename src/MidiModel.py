import torch
import torch.nn as nn


class EmotionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim,
                 hidden_dim, emotion_dim, num_emotions):
        super().__init__()
        self.note_embed = nn.Embedding(vocab_size, embed_dim)
        self.emotion_embed = nn.Embedding(num_emotions, emotion_dim)
        self.lstm = nn.LSTM(embed_dim + emotion_dim,
                            hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, notes, emotions):
        note_emb = self.note_embed(notes)
        emotion_emb = self.emotion_embed(emotions).unsqueeze(1)
        emotion_emb = emotion_emb.repeat(1, note_emb.size(1), 1)
        x = torch.cat([note_emb, emotion_emb], dim=2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
