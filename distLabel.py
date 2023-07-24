IPA = 4
NE_CLASS = 3
BATCH_SIZE = 100
EMB_DIM = 24
N_LAYERS = 2
HIDDEN_DIM = 64
NUM_CLASSES = 3
NUM_EPOCHS = 20
lr = 0.01
DROPOUT = 0.1


import csv
import epitran.vector
import panphon2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# df = pd.read_csv("ne_30k_emb.csv", index_col=0)
# # print(df["ipa"].values.astype(np.float32))
# X = torch.Tensor(df["ipa"].values)
# y = torch.Tensor(df["y"].values)


vwis = epitran.vector.VectorsWithIPASpace('eng-Latn', ['eng-Latn'])
ft = panphon2.FeatureTable()


with open("ne_ipa_pilot.csv") as file:
    reader = csv.reader(file)
    i = 0
    X = []
    y = []
    seq_lengths = []
    for rec in reader:
        i += 1
        if i == 1:
            continue
        # if i == 202:
        #     break
        if i % 10000 == 0: print(i)
        ipa = rec[IPA]
        neClass = 0
        if rec[NE_CLASS] == 'PER': neClass = 0
        elif rec[NE_CLASS] == 'LOC': neClass = 1
        elif rec[NE_CLASS] == 'ORG': neClass = 2
        emb = torch.tensor([[vec] for vec in ft.word_to_vectors(ipa)])
        # print(orth, rec[neClass])
        # emb = torch.tensor([([list(elem[5])]) for elem in segs])
        # emb = torch.tensor([([0 if j < 0 else j for j in list(elem[5])]) for elem in segs])
        X.append(emb)
        y.append(torch.tensor([neClass]))
        # seq_lengths.append(len(orth))

torch.save({"X": X, "y": y}, 'shit.pt')
# X = torch.load("gazetteer.pt")["X"]
# y = torch.load("gazetteer.pt")["y"]
# # print(X[:5])
# # print(y[:5])


# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)


# class NEDataset(Dataset):
#     def __init__(self, X, Y):
#         self.seq_lengths = [len(elem) for elem in X]
#         self.X = pad_sequence(X, batch_first=True)
#         self.y = Y
        
#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx], self.seq_lengths[idx]

# train_ds = NEDataset(X_train, y_train)
# valid_ds = NEDataset(X_valid, y_valid)

# def train_model(model, epochs=10, lr=0.001):
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = torch.optim.Adam(parameters, lr=lr)
#     for i in range(epochs):
#         model.train()
#         sum_loss = 0.0
#         total = 0
#         for x, y, seq_lengths in train_dl:
#             x = x.long().to(device)
#             y = y.long().to(device)
#             y_pred = model(x, seq_lengths)
#             optimizer.zero_grad()
#             loss = F.cross_entropy(y_pred, y)
#             loss.backward()
#             optimizer.step()
#             sum_loss += loss.item()*y.shape[0]
#             total += y.shape[0]
#         val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
#         print("Epoch %d: train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (i, sum_loss/total, val_loss, val_acc, val_rmse))

# def validation_metrics (model, valid_dl):
#     model.eval()
#     correct = 0
#     total = 0
#     sum_loss = 0.0
#     sum_rmse = 0.0
#     for x, y, l in valid_dl:
#         x = x.long().to(device)
#         y = y.long().to(device)
#         y_hat = model(x, l)
#         loss = F.cross_entropy(y_hat, y)
#         pred = torch.max(y_hat, 1)[1]
#         correct += (pred == y).float().sum()
#         total += y.shape[0]
#         sum_loss += loss.item()*y.shape[0]
#         sum_rmse += np.sqrt(mean_squared_error(pred.cpu(), y.cpu().unsqueeze(-1)))*y.cpu().shape[0]
#     return sum_loss/total, correct/total, sum_rmse/total

# train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# val_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)

# class LSTM_variable_input(nn.Module) :
#     def __init__(self, embedding_dim, hidden_dim) :
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(embedding_dim,
#                             hidden_dim,
#                             num_layers=N_LAYERS,
#                             bidirectional=True,
#                             dropout=DROPOUT,
#                             batch_first=True)
#         self.linear = nn.Linear(hidden_dim, NUM_CLASSES)
        
#     def forward(self, x, seq_lengths):
#         # print(x)
#         # print(seq_lengths)
#         x_pack = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
#         x_pack = x_pack.to(torch.float32)
#         out_pack, (ht, ct) = self.lstm(x_pack)
#         out = self.linear(ht[-1])
#         return out
    
# model = LSTM_variable_input(24, HIDDEN_DIM)
# model.to(device)

# train_model(model, epochs=NUM_EPOCHS, lr=lr)
# torch.save(model.state_dict(), './pilot_model_nonneg.pth')
