import torch
from tqdm import tqdm
import wandb
from utils import plot_losses

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_diff_loss = 0
    total_clip_loss = 0
    total_ce_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            energy_scores, protein_seq, peptide_seq = batch
            energy_scores = energy_scores.to(device)
            protein_tokens = protein_seq.to(device)
            peptide_tokens = peptide_seq.to(device)
            protein_onehot = F.one_hot(protein_tokens, num_classes=len(model.esm_model.alphabet)).float()

            protein_embedding = model.esm_model(protein_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            peptide_embedding = model.esm_model(peptide_tokens, repr_layers=[33], return_contacts=False)["representations"][33]

            epitope_latent = model.refined_representation(protein_tokens, (energy_scores <= -1).float())

            t = torch.randint(0, model.num_steps, (protein_embedding.shape[0],), device=device).long()
            loss, clip_loss, ce_loss, diff_loss = model.p_losses(peptide_embedding, protein_embedding, epitope_latent, t, peptide_tokens)

            total_loss += loss.item()
            total_clip_loss += clip_loss.item()
            total_ce_loss += ce_loss.item()
            total_diff_loss += (loss - clip_loss - ce_loss).item()

    avg_loss = total_loss / len(dataloader)
    avg_diff_loss = total_diff_loss / len(dataloader)
    avg_clip_loss = total_clip_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)

    return avg_loss, avg_diff_loss, avg_clip_loss, avg_ce_loss

def train(model, train_loader, val_loader, optimizer, num_epochs, device):
    wandb.init(project="protein_binding_diffusion", entity="vskavi2003")
    wandb.config.update({
        "learning_rate": optimizer.param_groups[0]['lr'],
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size
    })

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_diff_loss = 0
        total_train_clip_loss = 0
        total_train_ce_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            energy_scores, protein_seq, peptide_seq = batch
            energy_scores = energy_scores.to(device)
            protein_tokens = protein_seq.to(device)
            peptide_tokens = peptide_seq.to(device)

            protein_onehot = F.one_hot(protein_tokens, num_classes=len(model.esm_model.alphabet)).float()

            protein_embedding = model.esm_model(protein_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            peptide_embedding = model.esm_model(peptide_tokens, repr_layers=[33], return_contacts=False)["representations"][33]

            epitope_latent = model.refined_representation(protein_tokens, (energy_scores <= -1).float())

            if random.random() < 0.1:  # 10% of the time, remove antigen conditioning
                protein_embedding = torch.zeros_like(protein_embedding)
                epitope_latent = torch.zeros_like(epitope_latent)

            t = torch.randint(0, model.num_steps, (protein_embedding.shape[0],), device=device).long()
            loss, clip_loss, ce_loss, diff_loss = model.p_losses(peptide_embedding, protein_embedding, epitope_latent, t, peptide_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_clip_loss += clip_loss.item()
            total_train_ce_loss += ce_loss.item()
            total_train_diff_loss += (loss - clip_loss - ce_loss).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_diff_loss = total_train_diff_loss / len(train_loader)
        avg_train_clip_loss = total_train_clip_loss / len(train_loader)
        avg_train_ce_loss = total_train_ce_loss / len(train_loader)

        val_loss, val_diff_loss, val_clip_loss, val_ce_loss = validate(model, val_loader, device)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_diff_loss": avg_train_diff_loss,
            "train_clip_loss": avg_train_clip_loss,
            "train_ce_loss": avg_train_ce_loss,
            "val_loss": val_loss,
            "val_diff_loss": val_diff_loss,
            "val_clip_loss": val_clip_loss,
            "val_ce_loss": val_ce_loss
        })

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses
