import csv
import os
import random
import time
import numpy as np
import math
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import pandas as pd

from dataset import DrugTargetDataset
import metrics as EM

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

##########################################################################
"""Basic Settings"""


dataset = 'Davis'
drug_maxlen = 85  # Modify according to the dataset
target_maxlen = 1200  # Modify according to the dataset

# Set up result directory
result_path = f'runs/{dataset}'

os.makedirs(result_path, exist_ok=True)
csv_log = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 32  # Modify according to the dataset
accumulation_steps = 8  # Modify according to the dataset
learning_rate = 0.001
num_epochs = 600

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


        self.encoderD = Encoder(45, ST_size)  # SMILES Transformer vocabulary Size = 45
        self.encoderT = Encoder(33, ESM_size)  # ESM2 vocabulary Size = 33
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

##########################################################################
"""train & validation"""

kfold = KFold(n_splits=5, shuffle=False)

train_dataset = DrugTargetDataset(dataset=dataset, drug_maxlen=drug_maxlen, target_maxlen=target_maxlen, mode='train')




run_fold = 0  # Set to 0-4 to run specific fold

fold_result_path = os.path.join(result_path, f'fold_{fold}')
os.makedirs(fold_result_path, exist_ok=True)

if csv_log:
    result_file = os.path.join(fold_result_path, 'TrainingLog.csv')


    log_columns = ['fold', 'epoch', 'lr', 'epoch_time', 'train_loss', 'val_loss', 
                       'train_ci', 'val_ci', 'train_rm2', 'val_rm2',]
    
    with open(result_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=log_columns)
        writer.writeheader()


for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_dataset)))):
    if fold != run_fold:
        continue
    model = Transformer().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=30, verbose=False)



    print(f"################ fold {fold+1} train starts! ################\n")
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True
    )

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start = time.time()

        #### train ####
        model.train()
        total_train_loss = 0.0
        train_real_affi = []
        train_pred_affi = []
        for train_batch_idx, (drug_seqs, drug_pretraineds, target_seqs, target_pretraineds, labels) in enumerate(train_loader):
            drug_seqs = drug_seqs.to(device, non_blocking=True)
            target_seqs = target_seqs.to(device, non_blocking=True)
            drug_pretraineds = drug_pretraineds.to(device, non_blocking=True).float()
            target_pretraineds = target_pretraineds.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).squeeze()

            outputs, _, _, _, _ = model(drug_seqs, target_seqs, drug_pretraineds, target_pretraineds)

            train_loss = criterion(outputs, labels)
            
            total_train_loss += train_loss.item()

            train_loss.backward()

            train_real_affi.extend(labels.detach().cpu().tolist())
            train_pred_affi.extend(outputs.detach().cpu().tolist())

            if ((train_batch_idx + 1) % accumulation_steps) == 0 or (train_batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
        
        # train_MSE = EM.get_MSE(train_real_affi, train_pred_affi)
        train_CI = EM.get_ci(train_real_affi, train_pred_affi)
        train_rm2 = EM.get_rm2(train_real_affi, train_pred_affi)

        average_train_loss = total_train_loss / len(train_loader)

        #### val ####
        model.eval()
        total_val_loss = 0.0
        val_real_affi = []
        val_pred_affi = []
        with torch.no_grad():
            for drug_seqs, drug_pretraineds, target_seqs, target_pretraineds, labels in val_loader:
                drug_seqs = drug_seqs.to(device, non_blocking=True)
                target_seqs = target_seqs.to(device, non_blocking=True)
                drug_pretraineds = drug_pretraineds.to(device, non_blocking=True).float()
                target_pretraineds = target_pretraineds.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).squeeze()

                outputs, _, _, _, _ =  model(drug_seqs, target_seqs, drug_pretraineds, target_pretraineds)

                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                val_real_affi.extend(labels.detach().cpu().tolist())
                val_pred_affi.extend(outputs.detach().cpu().tolist())

        # val_MSE = EM.get_MSE(val_real_affi, val_pred_affi)
        val_CI = EM.get_ci(val_real_affi, val_pred_affi)
        val_rm2 = EM.get_rm2(val_real_affi, val_pred_affi)

        average_val_loss = total_val_loss / len(val_loader)

        scheduler.step(average_val_loss)
        
        early_stopping(average_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        end = time.time()

        print(f"Fold {fold +1}, Epoch {epoch + 1}/{num_epochs}, Lr: {optimizer.param_groups[0]['lr']}, "
        f"Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, "
        f"Time: {end - start:.2f} seconds")

        model_path_train = os.path.join(fold_result_path, 'best_train_model.pth')
        model_path_val = os.path.join(fold_result_path, 'best_val_model.pth')
        best_train_loss, best_val_loss, train_model_updated, val_model_updated = save_model(
            model, optimizer, epoch+1, average_train_loss, average_val_loss, 
            best_train_loss, best_val_loss, model_path_train, model_path_val
        )

        if csv_log:
            with open(result_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=log_columns)
                writer.writerow({
                    'fold' : fold,
                    'epoch': epoch + 1,
                    'lr': optimizer.param_groups[0]['lr'],
                    'epoch_time': end - start,
                    'train_loss': average_train_loss,
                    'val_loss': average_val_loss,
                    'train_ci': train_CI,
                    'val_ci': val_CI,
                    'train_rm2': train_rm2,
                    'val_rm2': val_rm2
                })
    
    final_model_path = os.path.join(fold_result_path, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, final_model_path)

##########################################################################
"""test"""
test_dataset = DrugTargetDataset(dataset=dataset, drug_maxlen=drug_maxlen, target_maxlen=target_maxlen, mode='test')

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def test_model(model_path, save_csv_path=None):
    model = Transformer().to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['loss']
    model.eval()

    total_test_loss = 0.0
    test_real_affi = []
    test_pred_affi = []
    
    criterion = nn.MSELoss(reduction='mean')
    
    with torch.no_grad():
        for drug_seqs, drug_pretraineds, target_seqs, target_pretraineds, labels in test_loader:
            drug_seqs = drug_seqs.to(device, non_blocking=True)
            target_seqs = target_seqs.to(device, non_blocking=True)
            drug_pretraineds = drug_pretraineds.to(device, non_blocking=True).float()
            target_pretraineds = target_pretraineds.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).squeeze()

            outputs, _, _, _, _ = model(drug_seqs, target_seqs, drug_pretraineds, target_pretraineds)
            test_loss = criterion(outputs, labels)
            total_test_loss += test_loss.item()
            
            test_real_affi.extend(labels.detach().cpu().tolist())
            test_pred_affi.extend(outputs.detach().cpu().tolist())
    
    average_test_loss = total_test_loss / len(test_loader)
    
    # compute metrics

    test_MSE = EM.get_MSE(test_real_affi, test_pred_affi)
    test_CI = EM.get_ci(test_real_affi, test_pred_affi)
    test_rm2 = EM.get_rm2(test_real_affi, test_pred_affi)
    
    print(f"Test Results for {model_path} saved in epoch {epoch} with train loss {train_loss} \ntest Loss: {average_test_loss:.4f}, MSE: {test_MSE:.4f}, CI: {test_CI:.4f}, RM2: {test_rm2:.4f}")

    # save predicted and real affinity values to csv
    if save_csv_path is not None:
        df = pd.DataFrame({
            'Real_Affinity': test_real_affi,
            'Predicted_Affinity': test_pred_affi
        })
        df.to_csv(save_csv_path, index=False)
        print(f"Predicted and actual values have been saved to: {save_csv_path}")

model_paths = [
    os.path.join(fold_result_path, 'best_train_model.pth'),
    os.path.join(fold_result_path, 'best_val_model.pth'),
    os.path.join(fold_result_path, 'final_model.pth')
]

for model_path in model_paths:
    test_model(model_path, save_csv_path=model_path.replace('.pth', '.csv'))