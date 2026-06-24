import json
import re
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

from smiles_transformer.pretrain_trfm import TrfmSeq2seq
from smiles_transformer.build_vocab import WordVocab
import __main__
__main__.WordVocab = WordVocab

class SMILESTransformer(nn.Module):
    def __init__(self, vocab_path, model_path, model_dim=256, num_layers=4):
        super(SMILESTransformer, self).__init__()

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # vocab
        self.vocab = WordVocab.load_vocab(vocab_path)

        # model
        self.model = TrfmSeq2seq(len(self.vocab), model_dim, len(self.vocab), num_layers)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # tokenizer
        self.TOKEN_PATTERN = re.compile(
            r'(\[|\]|Br|Cl|Si|Na|B|P|I|K|C|c|N|n|O|S|F|P|I|B|Na|Si|Se|K|#|=|/|\\|\+|-|\(|\)|\.|:|@|\?|>|\*|\$|%)'
        )

    def smiles_to_indices(self, smiles):
        indices = []
        for smile in smiles:
            tokens = self.TOKEN_PATTERN.findall(smile)
            indices.append([self.vocab.stoi.get(token, self.vocab.unk_index) for token in tokens])
        return indices

    def indices_to_tensor(self, indices, max_len=None):
        if max_len is None:
            max_len = max(len(idx) for idx in indices)

        tensor = torch.zeros((max_len, len(indices)), dtype=torch.long)

        for i, idx in enumerate(indices):
            tensor[:len(idx), i] = torch.tensor(idx, dtype=torch.long)

        return tensor.to(self.device)

    def forward(self, smiles_list):
        indices = self.smiles_to_indices(smiles_list)
        inputs = self.indices_to_tensor(indices)

        with torch.no_grad():
            predictions = self.model(inputs, return_hidden=True)

        return predictions


### ================== main ================== ###
if __name__ == "__main__":
    data_path = 'data'

    smiles_transformer = SMILESTransformer(
        vocab_path='smiles_transformer/vocab.pkl',
        model_path='smiles_transformer/trfm_12_23000.pkl',
        model_dim=256,
        num_layers=4
    )

    dataset = 'Davis'
    maxlen = 85

    file_path = f"data/{dataset}.txt"
    df = pd.read_csv(file_path, sep=" ")

    drug_df = df.iloc[:, [0, 2]].drop_duplicates()
    key_drugs = drug_df.iloc[:, 0].tolist()
    seq_drugs = drug_df.iloc[:, 1].tolist()

    print("Total drugs:", len(key_drugs))

    save_dir = os.path.join(data_path, dataset)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "drug_ST.pt")
    print("Saving to:", save_path)

    all_embeddings = {}

    for idx, seq in zip(key_drugs, seq_drugs):

        output = smiles_transformer([seq]).squeeze(1)  # (seq_len, dim)

        seq_len = output.shape[0]

        if seq_len > maxlen:
            output = output[:maxlen]
        elif seq_len < maxlen:
            pad_size = maxlen - seq_len
            output = F.pad(output, (0, 0, 0, pad_size), mode='constant', value=0)

        # Convert to float16 and move to CPU
        output = output.to(torch.float16).cpu()

        all_embeddings[str(idx)] = output

        print(f"Processed drug {idx}")

    # Save as .pt
    torch.save(all_embeddings, save_path)

    print("All drugs saved successfully.")