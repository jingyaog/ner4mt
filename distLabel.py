ORTH = 2
NE_CLASS = 4
BATCH_SIZE = 100
EMB_DIM = 24
HIDDEN_DIM = 10
NUM_CLASSES = 3
NUM_EPOCHS = 30
lr = 0.01


import csv
import epitran.vector
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


vwis = epitran.vector.VectorsWithIPASpace('eng-Latn', ['eng-Latn'])

with open("ne_pilot.csv") as file:
    reader = csv.reader(file)
    i = 0
    X = []
    y = []
    seq_lengths = []
    for rec in reader:
        i += 1
        if i == 1:
            continue
        # if i == 5:
        #     break
        if i % 100 == 0: print(i)
        orth = rec[ORTH]
        neClass = 0
        if rec[NE_CLASS] == 'PER': neClass = 0
        elif rec[NE_CLASS] == 'LOC': neClass = 1
        elif rec[NE_CLASS] == 'ORG': neClass = 2
        segs = vwis.word_to_segs(orth)
        # print(orth, rec[neClass])
        emb = torch.tensor([(list(elem[5])) for elem in segs])
        X.append(emb)
        y.append(neClass)
        seq_lengths.append(len(orth))
        
# X = pad_sequence(X, batch_first=True)
# print(y)
# print(seq_lengths)

# seq_lengths = torch.tensor(seq_lengths)
# seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
# padded_X = pad_sequence(X, batch_first=True)
# padded_X = padded_X[perm_idx]
# packed_X = pack_padded_sequence(padded_X, seq_lengths.cpu().numpy(), batch_first=True)
# print(seq_lengths.cpu().numpy())
# print(padded_X)
# print(packed_X)
# print(len(packed_X))
# print(len(y))


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


class NEDataset(Dataset):
    def __init__(self, X, Y):
        self.seq_lengths = [len(elem) for elem in X]
        self.X = pad_sequence(X, batch_first=True)
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.seq_lengths[idx]

train_ds = NEDataset(X_train, y_train)
valid_ds = NEDataset(X_valid, y_valid)

# print(valid_ds[0])

def train_model(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, seq_lengths in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, seq_lengths)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        print(correct)
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)

class LSTM_variable_input(nn.Module) :
    def __init__(self, embedding_dim, hidden_dim) :
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, NUM_CLASSES)
        
    def forward(self, x, seq_lengths):
        x_pack = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x_pack = x_pack.to(torch.float32)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out
    
model = LSTM_variable_input(24, HIDDEN_DIM)

train_model(model, epochs=NUM_EPOCHS, lr=lr)