import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import re

def preprocess_snp_data(file_path):
    snp_df = pd.read_csv(file_path)

    def transform_energy_scores(energy_scores):
        transformed_scores = []
        for score in energy_scores:
            score = re.sub(r'[\s\n]+', ',', score)
            score = re.sub(r'\[\s*,', '[', score)
            score = re.sub(r'^[\s,]+', '', score)
            transformed_scores.append(score)
        return transformed_scores

    snp_df['energy_scores'] = transform_energy_scores(snp_df['energy_scores'])
    snp_df['energy_scores_lengths'] = snp_df['energy_scores'].apply(
        lambda x: x.count(',') + 1 - (1 if x.startswith(',') else 0)
    )

    snp_df['peptide_source_RCSB_lengths'] = snp_df['peptide_source_RCSB'].apply(len)
    snp_df['protein_RCSB_lengths'] = snp_df['protein_RCSB'].apply(len)
    snp_df['protein_derived_seq_length'] = snp_df['protein_derived_sequence'].apply(len)
    snp_df['peptide_derived_seq_length'] = snp_df['peptide_derived_sequence'].apply(len)

    return snp_df

def filter_datasets(dataset):
    return dataset[dataset['protein_RCSB'] != dataset['peptide_source_RCSB']]

class ProteinInteractionDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.mismatched_lengths = 0
        self.total_samples = len(dataframe)
        self.check_lengths()

    def check_lengths(self):
        for idx in range(self.total_samples):
            row = self.dataframe.iloc[idx]
            peptide_seq = row['peptide_derived_sequence']
            energy_scores = row['energy_scores']

            energy_scores = re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', energy_scores)
            energy_scores = [float(score) for score in energy_scores]

            if len(energy_scores) != len(peptide_seq):
                self.mismatched_lengths += 1

        print(f"Total samples: {self.total_samples}")
        print(f"Mismatched lengths: {self.mismatched_lengths}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        peptide_seq = row['peptide_derived_sequence']
        protein_seq = row['protein_derived_sequence']
        energy_scores = row['energy_scores']

        energy_scores = re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', energy_scores)
        energy_scores = [float(score) for score in energy_scores]
        energy_scores = self.one_hot_encode_energy_scores(energy_scores)

        energy_scores = torch.tensor(energy_scores, dtype=torch.float32)

        return energy_scores, peptide_seq, protein_seq

    @staticmethod
    def one_hot_encode_energy_scores(scores):
        return [1 if score <= -1 else 0 for score in scores]

def load_and_preprocess_data(train_path, val_path, test_path, subset_size=None):
    train_snp = preprocess_snp_data(train_path)
    val_snp = preprocess_snp_data(val_path)
    test_snp = preprocess_snp_data(test_path)

    train_snp = filter_datasets(train_snp)
    val_snp = filter_datasets(val_snp)
    test_snp = filter_datasets(test_snp)

    if subset_size:
        train_snp = train_snp[:subset_size[0]]
        val_snp = val_snp[:subset_size[1]]
        test_snp = test_snp[:subset_size[2]]

    all_seqs = pd.concat([
        train_snp['peptide_derived_sequence'], train_snp['protein_derived_sequence'],
        val_snp['peptide_derived_sequence'], val_snp['protein_derived_sequence'],
        test_snp['peptide_derived_sequence'], test_snp['protein_derived_sequence']
    ])
    max_length = max(len(seq) for seq in all_seqs)

    train_dataset = ProteinInteractionDataset(train_snp)
    val_dataset = ProteinInteractionDataset(val_snp)
    test_dataset = ProteinInteractionDataset(test_snp)

    return train_dataset, val_dataset, test_dataset, max_length

def custom_collate_fn(batch, esm_model):
    energy_scores, protein_seqs, peptide_seqs = zip(*batch)
    batch_converter = esm_model.alphabet.get_batch_converter()
    _, _, protein_tokens = batch_converter([(i, seq) for i, seq in enumerate(protein_seqs)])
    _, _, peptide_tokens = batch_converter([(i, seq) for i, seq in enumerate(peptide_seqs)])

    padded_energy_scores = [F.pad(torch.tensor(score, dtype=torch.float32), (1, 1), value=0) for score in energy_scores]

    max_seq_length = max(protein_tokens.size(1), peptide_tokens.size(1))

    padded_energy_scores = torch.stack([
        pad_or_truncate(score.unsqueeze(0), max_seq_length, pad_value=0).squeeze(0)
        for score in padded_energy_scores
    ])

    padded_protein_tokens = pad_or_truncate(protein_tokens, max_seq_length, pad_value=esm_model.alphabet.padding_idx)
    padded_peptide_tokens = pad_or_truncate(peptide_tokens, max_seq_length, pad_value=esm_model.alphabet.padding_idx)

    return padded_energy_scores, padded_protein_tokens, padded_peptide_tokens

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, esm_model):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, 
                              collate_fn=lambda x: custom_collate_fn(x, esm_model))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, 
                            collate_fn=lambda x: custom_collate_fn(x, esm_model))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, 
                             collate_fn=lambda x: custom_collate_fn(x, esm_model))
    return train_loader, val_loader, test_loader
