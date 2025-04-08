import torch
from src.MidiModel import EmotionLSTM
from src.MidiUtils import sequence_to_midi, load_model


def generate(model, seed, emotion_id, length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_seq = torch.tensor(seed[-32:], dtype=torch.long).unsqueeze(0).to(device)
    emotion_tensor = torch.tensor([emotion_id], dtype=torch.long).to(device)
    result = seed[:]

    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq, emotion_tensor)
            _, predicted = torch.max(output, dim=1)
            result.append(predicted.item())
            input_seq = torch.cat([input_seq[:, 1:], predicted.unsqueeze(0)], dim=1)

    return result

# Example usage:
# model = load_model(EmotionLSTM, "outputs/emotion_lstm.pth", vocab_size, 64, 128, 16, 4).to("cuda")
# result_seq = generate(model, seed_sequence, emotion_id)
# sequence_to_midi(result_seq, int_to_note, output_path="outputs/generated.mid")
