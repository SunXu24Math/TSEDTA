import json
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data_path = 'data'

# vocabulary for ESM2
vocab_esm = {'<pad>': 0, '<cls>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 
             'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 
             'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 
             'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 
             'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 
             'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, 
             '<null_1>': 31, '<mask>': 32}
vocab_esm_size = 33

# vocabulary for SMILES Transformer
vocab_ST = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4, 'c': 5, 
             'C': 6, '(': 7, ')': 8, 'O': 9, '=': 10, 
             '1': 11, 'N': 12, '2': 13, '3': 14, 'n': 15, 
             '4': 16, '@': 17, '[': 18, ']': 19, 'H': 20, 
             'F': 21, '5': 22, 'S': 23, '\\': 24, 'Cl': 25, 
             's': 26, '6': 27, 'o': 28, '+': 29, '-': 30, 
             '#': 31, '/': 32, '.': 33, 'Br': 34, '7': 35, 
             'P': 36, 'I': 37, '8': 38, 'Na': 39, 'B': 40, 
             'Si': 41, 'Se': 42, '9': 43, 'K': 44}
vocab_ST_size = 45

TOKEN_PATTERN = re.compile(r'(\[|\]|Br|Cl|Si|Na|B|P|I|K|C|c|N|n|O|S|F|P|I|B|Na|Si|Se|K|#|=|/|\\|\+|-|\(|\)|\.|:|@|\?|>|\*|\$|%)')

def tokenize_smiles(smiles):
    """使用给定 TOKEN_PATTERN 进行 SMILES 分词"""
    return [match.group(0) for match in TOKEN_PATTERN.finditer(smiles)]

def encode_smiles_sequence(seq, vocab, maxlen):
    tokens = tokenize_smiles(seq)
    ids = []
    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab['<unk>'])

    ids = ids[:maxlen]
    # padding
    if len(ids) < maxlen:
        ids.extend([vocab['<pad>']] * (maxlen - len(ids)))
    return ids

def encode_protein_sequence(seq, vocab, maxlen):
    ids = []
    for aa in seq:
        if aa in vocab:
            ids.append(vocab[aa])
        else:
            ids.append(vocab['<unk>'])

    ids = ids[:maxlen]
    # padding
    if len(ids) < maxlen:
        ids.extend([vocab['<pad>']] * (maxlen - len(ids)))
    return ids

# ==========================================================
# Master data processing function
# ==========================================================
def process_data(dataset, drug_maxlen, target_maxlen, random_state=None):
    file_path = f"data/{dataset}.txt"
    df = pd.read_csv(file_path, sep=" ")
    
    key_drugs = df.iloc[:, 0].tolist()
    key_targets = df.iloc[:, 1].tolist()
    seq_drugs = df.iloc[:, 2].tolist()
    seq_targets = df.iloc[:, 3].tolist()
    affinities = df.iloc[:, 4].tolist()

    dir_path = os.path.join(data_path, dataset)
    os.makedirs(dir_path, exist_ok=True)

    print("Encoding SMILES...")
    encoded_drugs = [
        encode_smiles_sequence(seq, vocab_ST, drug_maxlen)
        for seq in seq_drugs
    ]
    print("Encoding Protein sequences...")
    encoded_targets = [
        encode_protein_sequence(seq, vocab_esm, target_maxlen)
        for seq in seq_targets
    ]
    
    drugs_dict = {key: seq for key, seq in zip(key_drugs, encoded_drugs)}
    drug_json = os.path.join(dir_path, 'drugs.json')
    with open(drug_json, 'w') as f:
        json.dump(drugs_dict, f)

    targets_dict = {key: seq for key, seq in zip(key_targets, encoded_targets)}
    target_json = os.path.join(dir_path, 'targets.json')
    with open(target_json, 'w') as f:
        json.dump(targets_dict, f)

    split_train_test(key_drugs, key_targets, affinities, dir_path, test_size=1/6, random_state=random_state)


def split_train_test(key_drugs, key_targets, affinities, dir_path, test_size=1/6, random_state=None):
    data = pd.DataFrame({
        'key_drug': key_drugs,
        'key_target': key_targets,
        'affinity': affinities
    })

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    train_data.to_csv(os.path.join(dir_path, 'train_mapping.csv'), index=False)
    test_data.to_csv(os.path.join(dir_path, 'test_mapping.csv'), index=False)

if __name__ == '__main__':
    process_data('Metz', drug_maxlen=80, target_maxlen=1000, random_state=42)

