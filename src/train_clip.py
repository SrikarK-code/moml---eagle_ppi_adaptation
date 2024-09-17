import torch
from tqdm import tqdm
import wandb

def train_clip(model, esm_model, train_loader, val_loader, optimizer, num_epochs, device):
    wandb.init(project="protein_binding_clip", entity="vskavi2003")
    wandb.config.update({
        "learning_rate": optimizer.param_groups[0]['lr'],
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size
    })

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"CLIP Epoch {epoch+1}/{num_epochs}"):
            energy_scores, protein_seq, peptide_seq = batch
            energy_scores = energy_scores.to(device)
            protein_tokens = protein_seq.to(device)
            peptide_tokens = peptide_seq.to(device)
            protein_onehot = F.one_hot(protein_tokens, num_classes=len(esm_model.alphabet)).float()

            with torch.no_grad():
                protein_embedding = esm_model(protein_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
                peptide_embedding = esm_model(peptide_tokens, repr_layers=[33], return_contacts=False)["representations"][33]

            _, _, clip_loss = model(peptide_embedding, protein_embedding)

            optimizer.zero_grad()
            clip_loss.backward()
            optimizer.step()

            total_train_loss += clip_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                energy_scores, protein_seq, peptide_seq = batch
                energy_scores = energy_scores.to(device)
                protein_tokens = protein_seq.to(device)
                peptide_tokens = peptide_seq.to(device)
                protein_onehot = F.one_hot(protein_tokens, num_classes=len(esm_model.alphabet)).float()

                protein_embedding = esm_model(protein_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
                peptide_embedding = esm_model(peptide_tokens, repr_layers=[33], return_contacts=False)["representations"][33]

                _, _, val_clip_loss = model(peptide_embedding, protein_embedding)
                total_val_loss += val_clip_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch+1,
            "clip_train_loss": avg_train_loss,
            "clip_val_loss": avg_val_loss
        })

        print(f"CLIP Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
