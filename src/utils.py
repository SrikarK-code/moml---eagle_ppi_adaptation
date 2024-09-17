import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def pad_or_truncate(tensor, target_length, pad_value=0):
    current_length = tensor.size(1)
    if current_length < target_length:
        padding = torch.full((tensor.size(0), target_length - current_length, *tensor.size()[2:]), pad_value, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)
    else:
        return tensor[:, :target_length]

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    return plt.gcf()

def generate_random_protein_binders(model, num_samples=5, seq_length=100):
    device = next(model.parameters()).device
    protein_seq = ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(seq_length))
    epitope_start = random.randint(0, seq_length - 10)
    epitope_end = epitope_start + random.randint(5, 10)
    epitope_seq = protein_seq[epitope_start:epitope_end]
    generated_binders = model.sample(protein_seq, epitope_seq, num_samples=num_samples)
    return protein_seq, epitope_seq, generated_binders
