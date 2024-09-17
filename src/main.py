import torch
import esm
from model import RefinedLatentDiffusion
from data_loading import load_and_preprocess_data, create_data_loaders
from train_model import train
from train_clip import train_clip
from utils import plot_losses, generate_random_protein_binders
import wandb

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ESM model
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    for param in esm_model.parameters():
        param.requires_grad = False

    # Load and preprocess data
    train_dataset, val_dataset, test_dataset, max_length = load_and_preprocess_data(
        'training_dataset.csv',
        'validation_dataset.csv',
        'testing_dataset.csv',
        subset_size=(32, 16, 16)  # Adjust these numbers as needed
    )

    # Create data loaders
    batch_size = 2
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, esm_model
    )

    # Initialize model
    num_steps = 1000
    model = RefinedLatentDiffusion(esm_model, num_steps, device).to(device)

    # Train CLIP model
    clip_optimizer = torch.optim.Adam(model.clip_model.parameters(), lr=1e-5)
    train_clip(model.clip_model, esm_model, train_loader, val_loader, clip_optimizer, num_epochs=2, device=device)

    # Train Latent Diffusion model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, num_epochs=2, device=device)

    # Plot losses
    loss_plot = plot_losses(train_losses, val_losses)
    wandb.log({"loss_plot": wandb.Image(loss_plot)})

    # Save the trained model
    torch.save(model.state_dict(), 'protein_binder_model.pth')

    # Generate examples
    print("Generating protein binders for random cases:")
    for i in range(3):
        protein_seq, epitope_seq, generated_binders = generate_random_protein_binders(model)
        print(f"\nCase {i+1}:")
        print(f"Protein sequence: {protein_seq}")
        print(f"Epitope sequence: {epitope_seq}")
        print("Generated binders:")
        for j, binder in enumerate(generated_binders):
            print(f"Binder {j+1}: {binder}")

    wandb.finish()

if __name__ == "__main__":
    main()
