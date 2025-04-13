import torch

def generate(model, input_seq, emotion_label, length=50):
    device = next(model.parameters()).device  # 🔍 Detect model device (MPS or CPU)
    
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)  # (1, sequence_len)
    emotion_tensor = torch.tensor([emotion_label]).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq, emotion_tensor)
            next_note = output.argmax(-1).item()
            input_seq = torch.cat([input_seq, torch.tensor([[next_note]], device=device)], dim=1)

    return input_seq[0].tolist()
