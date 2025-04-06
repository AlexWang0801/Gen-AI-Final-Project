import torch
from src.MidiModel import EmotionLSTM
from src.MidiUtils import sequence_to_midi, load_model


def generate(model, seed, emotion_id, length=50):
    model.eval()
    result = seed[:]
    input_seq = torch.tensor(seed[-32:], dtype=torch.long).unsqueeze(0).to("cuda")
    emotion = torch.tensor([emotion_id]).to("cuda")

    for _ in range(length):
        out = model(input_seq, emotion)
        next_note = torch.argmax(out, dim=1).item()
        result.append(next_note)
        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_note]]).to("cuda")], dim=1)
    return result

# Example usage:
# model = load_model(EmotionLSTM, "outputs/emotion_lstm.pth", vocab_size, 64, 128, 16, 4).to("cuda")
# result_seq = generate(model, seed_sequence, emotion_id)
# sequence_to_midi(result_seq, int_to_note, output_path="outputs/generated.mid")
