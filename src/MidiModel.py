import torch
import torch.nn as nn
import math

# Define the LSTM model for music generation
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


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EmotionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, dim_feedforward,
                 num_encoder_layers, emotion_dim, num_emotions, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.note_embed = nn.Embedding(vocab_size, embed_dim)
        # Emotion Embedding
        # Use same embed_dim
        self.emotion_embed = nn.Embedding(num_emotions, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Input shape (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.note_embed.weight.data.uniform_(-initrange, initrange)
        self.emotion_embed.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        """Generates a causal mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, notes, emotions):
        """
        Args:
            notes: Tensor, shape [batch_size, seq_len] - Note indices
            emotions: Tensor, shape [batch_size] - Emotion indices

        Returns:
            Tensor, shape [batch_size, vocab_size] - Logits for the next note prediction
        """
        # 1. Embeddings
        note_emb = self.note_embed(notes) * math.sqrt(self.embed_dim) # Scale embedding (common practice)

        # --- Combine with Emotion Embedding ---
        # Add emotion embedding
        # Unsqueeze emotion embedding to match note_emb dimensions for broadcasting
        emotion_emb = self.emotion_embed(emotions).unsqueeze(1) # [batch_size, 1, embed_dim]
        combined_emb = note_emb + emotion_emb # Broadcasting adds emotion to each note position

        # 2. Add Positional Encoding
        pos_encoded_emb = self.pos_encoder(combined_emb) # Input: [batch, seq, feature]

        # 3. Generate Causal Mask
        # Mask should prevent attending to future tokens
        # Shape: [seq_len, seq_len]
        seq_len = notes.size(1)
        device = notes.device
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)

        # 4. Transformer Encoder
        # Input: [batch, seq, feature], Mask: [seq, seq]
        transformer_output = self.transformer_encoder(pos_encoded_emb, mask=causal_mask)
        # Output shape: [batch, seq, feature]

        # 5. Final Prediction Layer
        last_token_output = transformer_output[:, -1, :] # Shape: [batch_size, embed_dim]
        logits = self.fc_out(last_token_output) # Shape: [batch_size, vocab_size]

        return logits