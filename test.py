import csv
import os
import random
import time
import numpy as np
import math
import json
import re
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from dataset import DrugTargetDataset
import metrics as EM
from preprocess_drug import SMILESTransformer
from preprocess_protein import ESM


torch.set_num_threads(8)

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed = 42
seed_torch(seed=seed)


#########################################################################
"""Basic Settings"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dropout = 0.1
d_model = 128
d_ff = 512
d_k = d_v = 32
n_layers = 1
n_heads = 4

ST_size = 256
ESM_size = 1280

##########################################################################
"""model"""
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        self.encoderD = Encoder(45, ST_size)
        self.encoderT = Encoder(33, ESM_size)
        self.fc0 = nn.Sequential(
            nn.Linear(2*d_model, 8*d_model, bias=False),
            nn.LayerNorm(8*d_model),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8*d_model, 4*d_model, bias=False),
            nn.LayerNorm(4*d_model),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(4*d_model, 1, bias=False)


    def forward(self, input_Drugs, input_Tars, drug_pretrained, target_pretrained):
        enc_Drugs, enc_attnsD1, enc_attnsD2 = self.encoderD(input_Drugs, drug_pretrained)
        enc_Tars, enc_attnsT1, enc_attnsT2 = self.encoderT(input_Tars, target_pretrained)

        enc_Drugs_2D0 = torch.sum(enc_Drugs, dim=1)
        enc_Drugs_2D1 = enc_Drugs_2D0.squeeze()
        enc_Tars_2D0 = torch.sum(enc_Tars, dim=1)
        enc_Tars_2D1 = enc_Tars_2D0.squeeze()
        if enc_Drugs_2D1.dim() == 1:
            enc_Drugs_2D1 = enc_Drugs_2D1.unsqueeze(0)
        if enc_Tars_2D1.dim() == 1:
            enc_Tars_2D1 = enc_Tars_2D1.unsqueeze(0)
        fc = torch.cat((enc_Drugs_2D1, enc_Tars_2D1), 1)
        # enc_Drugs_2D = torch.mean(enc_Drugs, dim=1)
        # enc_Tars_2D = torch.mean(enc_Tars, dim=1)
        # fc = torch.cat((enc_Drugs_2D, enc_Tars_2D), 1)

        fc0 = self.fc0(fc)
        fc1 = self.fc1(fc0)
        fc2 = self.fc2(fc1)
        affi = fc2.squeeze()

        return affi, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2

class Encoder(nn.Module):
    def __init__(self, vocab_size, pretrain_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.adjust = nn.Sequential(
            nn.Linear(pretrain_size, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.stream1 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, embeddings):

        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        
        enc_self_attns1, enc_self_attns2 = [], []
        stream0 = self.adjust(embeddings)

        stream1 = stream0 + enc_outputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        return stream1, enc_self_attns1, enc_self_attns2

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output+residual) # [batch_size, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        
        ##residual, batch_size = input_Q, input_Q.size(0)
        batch_size, seq_len, model_len = input_Q.size()
        residual_2D = input_Q.view(batch_size*seq_len, model_len)
        residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                      2) # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                               1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output+residual), attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

def get_attn_pad_mask(seq_q, seq_k):
    # seq_q=seq_k: [batch_size, seq_len]

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [batch_size, len_q, len_k]


##########################################################################
"""auto"""
class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0):
        """
        Args:
            patience (int): if validation loss doesn't improve for this many epochs, training will be stopped.
            verbose (bool): if True, prints a message for each validation loss improvement.
            delta (float): minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.val_loss_min = val_loss
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.val_loss_min = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def save_model(model, optimizer, epoch, train_loss, val_loss, best_train_loss, best_val_loss, model_path_train, model_path_val):
    train_loss_updated = 0
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        train_loss_updated = 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, model_path_train)

    val_loss_updated = 0
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        val_loss_updated = 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, model_path_val)

    return best_train_loss, best_val_loss, train_loss_updated, val_loss_updated


def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss



##########################################################################
"""test"""

drug = "CCN(C)C(=O)OC1=CC=CC(=C1)[C@H](C)N(C)C"

protein = "MRPPQCLLHTPSLASPLLLLLLWLLGGGVGAEGREDAELLVTVRGGRLRGIRLKTPGGPVSAFLGIPFAEPPMGPRRFLPPEPKQPWSGVVDATTFQSVCYQYVDTLYPGFEGTEMWNPNRELSEDCLYLNVWTPYPRPTSPTPVLVWIYGGGFYSGASSLDVYDGRFLVQAERTVLVSMNYRVGAFGFLALPGSREAPGNVGLLDQRLALQWVQENVAAFGGDPTSVTLFGESAGAASVGMHLLSPPSRGLFHRAVLQSGAPNGPWATVGMGEARRRATQLAHLVGCPPGGTGGNDTELVACLRTRPAQVLVNHEWHVLPQESVFRFSFVPVVDGDFLSDTPEALINAGDFHGLQVLVGVVKDEGSYFLVYGAPGFSKDNESLISRAEFLAGVRVGVPQVSDLAAEAVVLHYTDWLHPEDPARLREALSDVVGDHNVVCPVAQLAGRLAAQGARVYAYVFEHRASTLSWPLWMGVPHGYEIEFIFGIPLDPSRNYTAEEKIFAQRLMRYWANFARTGDPNEPRDPKAPQWPPYTAGAQQYVSLDLRPLEVRRGLRAQACAFWNRFLPKLLSATDTLDEAERQWKAEFHRWSSYMVHWKNQFDHYSKQDRCSDL"

dataset = 'Davis'
drug_maxlen = 85
protein_maxlen = 1200


#########################################################################
# preprocess drug and protein sequences
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

vocab_esm = {'<pad>': 0, '<cls>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 
             'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 
             'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 
             'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 
             'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 
             'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, 
             '<null_1>': 31, '<mask>': 32}
vocab_esm_size = 33

TOKEN_PATTERN = re.compile(r'(\[|\]|Br|Cl|Si|Na|B|P|I|K|C|c|N|n|O|S|F|P|I|B|Na|Si|Se|K|#|=|/|\\|\+|-|\(|\)|\.|:|@|\?|>|\*|\$|%)')
tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(drug)]
print("Tokenized SMILES:", tokens)
encoded_drug = []
for token in tokens:
    if token in vocab_ST:
        encoded_drug.append(vocab_ST[token])
    else:
        encoded_drug.append(vocab_ST['<unk>'])

encoded_drug = encoded_drug[:drug_maxlen]
# padding
if len(encoded_drug) < drug_maxlen:
    encoded_drug.extend([vocab_ST['<pad>']] * (drug_maxlen - len(encoded_drug)))
print("Encoded drug sequence:", encoded_drug)

encoded_protein = []
for aa in protein:
    if aa in vocab_esm:
        encoded_protein.append(vocab_esm[aa])
    else:
        encoded_protein.append(vocab_esm['<unk>'])

encoded_protein = encoded_protein[:protein_maxlen]
# padding
if len(encoded_protein) < protein_maxlen:
    encoded_protein.extend([vocab_esm['<pad>']] * (protein_maxlen - len(encoded_protein)))
print("Encoded protein sequence:", encoded_protein)


#######################################################################
# pretrained drug and protein features

smiles_transformer = SMILESTransformer(
    vocab_path='smiles_transformer/vocab.pkl',
    model_path='smiles_transformer/trfm_12_23000.pkl',
    model_dim=256,
    num_layers=4
)
pretrained_drug = smiles_transformer([drug]).squeeze(1)  # (seq_len, dim)
seq_len = pretrained_drug.shape[0]
if seq_len > drug_maxlen:
    pretrained_drug = pretrained_drug[:drug_maxlen]
elif seq_len < drug_maxlen:
    pad_size = drug_maxlen - seq_len
    pretrained_drug = F.pad(pretrained_drug, (0, 0, 0, pad_size), mode='constant', value=0)
print("Pretrained drug feature shape:", pretrained_drug.shape)

esm = ESM(device=device)
pretrained_protein = esm((None, protein))  # (seq_len, dim)
current_len = pretrained_protein.shape[0]
if current_len > protein_maxlen:
    pretrained_protein = pretrained_protein[:protein_maxlen]
elif current_len < protein_maxlen:
    pad_size = protein_maxlen - current_len
    pretrained_protein = F.pad(pretrained_protein, (0, 0, 0, pad_size), value=0)
print("Pretrained protein feature shape:", pretrained_protein.shape)



#######################################################################
# model inference

def test_model(model_path=f"models/{dataset}_model.pth"):

    # =========================
    # load model
    # =========================
    model = Transformer().to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Train loss: {checkpoint.get('loss', 'unknown')}")

    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # =========================
    # convert to tensor
    # =========================

    # sequence feature
    drug_seq_tensor = torch.tensor(
        encoded_drug,
        dtype=torch.long
    ).unsqueeze(0).to(device)   # (1, drug_maxlen)

    protein_seq_tensor = torch.tensor(
        encoded_protein,
        dtype=torch.long
    ).unsqueeze(0).to(device)   # (1, protein_maxlen)

    # pretrained feature
    drug_pretrained_tensor = pretrained_drug.unsqueeze(0).float().to(device)
    # (1, drug_maxlen, dim)

    protein_pretrained_tensor = pretrained_protein.unsqueeze(0).float().to(device)
    # (1, protein_maxlen, dim)

    print("drug_seq_tensor shape:", drug_seq_tensor.shape)
    print("protein_seq_tensor shape:", protein_seq_tensor.shape)

    print("drug_pretrained_tensor shape:", drug_pretrained_tensor.shape)
    print("protein_pretrained_tensor shape:", protein_pretrained_tensor.shape)

    # =========================
    # inference
    # =========================
    with torch.no_grad():

        outputs, _, _, _, _ = model(
            drug_seq_tensor,
            protein_seq_tensor,
            drug_pretrained_tensor,
            protein_pretrained_tensor
        )

    prediction = outputs.squeeze().item()

    print("\n==============================")
    print("Predicted affinity:", prediction)
    print("==============================\n")

    return prediction

pred = test_model(f"models/{dataset}_model.pth")



