import torch
import torch.nn.functional as F

def generate(model, input_seq, emotion_label, length=50):
    device = next(model.parameters()).device
    
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    emotion_tensor = torch.tensor([emotion_label]).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq, emotion_tensor)
            next_note = output.argmax(-1).item()
            input_seq = torch.cat([input_seq, torch.tensor([[next_note]], device=device)], dim=1)

    return input_seq[0].tolist()


def generate_using_transformer(model, input_seq_indices, emotion_label, seq_len, length=50, temperature=1.0, top_k=None):
    """Generates a sequence using the Transformer model."""
    device = next(model.parameters()).device
    model.eval()

    current_sequence = list(input_seq_indices) # Start with the seed sequence
    generated_indices = []

    with torch.no_grad():
        for _ in range(length):
            # Prepare the input: last seq_len tokens
            input_tensor_indices = current_sequence[-seq_len:]
            input_tensor = torch.tensor([input_tensor_indices], dtype=torch.long).to(device)
            emotion_tensor = torch.tensor([emotion_label], dtype=torch.long).to(device)
            logits = model(input_tensor, emotion_tensor)
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probabilities = F.softmax(logits, dim=-1)
            next_note_idx = torch.multinomial(probabilities, num_samples=1).item()

            current_sequence.append(next_note_idx)
            generated_indices.append(next_note_idx)

    return current_sequence