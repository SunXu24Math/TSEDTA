import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset


data_path = 'data'


class DrugTargetDataset(Dataset):
    def __init__(self, dataset, drug_maxlen, target_maxlen, mode='train'):
        self.drug_maxlen = drug_maxlen
        self.target_maxlen = target_maxlen

        mapping_csv = os.path.join(data_path, dataset, f"{mode}_mapping.csv")
        self.mapping = pd.read_csv(mapping_csv)

        # Original token sequence dictionary
        drug_json = os.path.join(data_path, dataset, 'drugs.json')
        with open(drug_json, 'r') as f:
            self.drugs_dict = json.load(f)

        target_json = os.path.join(data_path, dataset, 'targets.json')
        with open(target_json, 'r') as f:
            self.targets_dict = json.load(f)

        # =========================
        # Pre-trained drug embedding (.pt)
        # =========================
        drugs_pt_path = os.path.join(data_path, dataset, 'drug_ST.pt')
        self.drug_pretrained_dict = torch.load(drugs_pt_path, map_location='cpu')

        print(f"Loaded drug pretrained embeddings: {len(self.drug_pretrained_dict)} drugs")

        # =========================
        # Pre-trained target embedding (.pt)
        # =========================
        self.targets_ESM_dir = os.path.join(data_path, dataset, 'targets_ESM')

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        row = self.mapping.iloc[idx]

        key_drug = str(row['key_drug'])
        key_target = str(row['key_target'])
        affinity = row['affinity']

        # =========================
        # Original token sequence
        # =========================
        drug_seq = torch.tensor(
            self.drugs_dict[key_drug],
            dtype=torch.long
        )

        target_seq = torch.tensor(
            self.targets_dict[key_target],
            dtype=torch.long
        )

        # =========================
        # drug pretrained (float16)
        # =========================
        drug_pretrained = self.drug_pretrained_dict[key_drug]

        # =========================
        # target pretrained (.pt float16)
        # =========================
        target_pretrained_path = os.path.join(
            self.targets_ESM_dir,
            f"{key_target}.pt"
        )

        target_pretrained = torch.load(
            target_pretrained_path,
            map_location='cpu'
        )

        return (
            drug_seq,
            drug_pretrained,      # float16
            target_seq,
            target_pretrained,    # float16
            torch.tensor([affinity], dtype=torch.float32)
        )


if __name__ == '__main__':

    dataset = DrugTargetDataset(
        dataset='Metz',
        drug_maxlen=80,
        target_maxlen=1000,
        mode='test'
    )

    drug_seq, drug_pretrained, target_seq, target_pretrained, affinity = dataset[0]

    print(f"drug_seq shape: {drug_seq.shape}, dtype: {drug_seq.dtype}")
    print(f"drug_pretrained shape: {drug_pretrained.shape}, dtype: {drug_pretrained.dtype}")
    print(f"target_seq shape: {target_seq.shape}, dtype: {target_seq.dtype}")
    print(f"target_pretrained shape: {target_pretrained.shape}, dtype: {target_pretrained.dtype}")
    print(f"affinity shape: {affinity.shape}, dtype: {affinity.dtype}")