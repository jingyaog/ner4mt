# EMB_DIM = 24

import numpy as np
import pandas as pd
import panphon2
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
ft = panphon2.FeatureTable()
 

def isEnglish(s):
    if not isinstance(s, str): return False
    if s.replace('.', '').isdigit(): return False
    if len(s) <= 1: return False
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def isValidIPA(ipa):
    if not isinstance(ipa, str) or len(ipa) <= 0: return False
    return not any(i.isdigit() for i in ipa)
    
df = pd.read_csv("ne_ipa.csv")
# df = df.loc[df["ipa"].apply(isValidIPA)]
# df = df.loc[df["norm_name"].apply(isEnglish)]
X = df["ipa"]
y = df["neClass"]
print(len(X))
print(len(y))

# all_letters = set()
# for ipa in X:
#     for char in ipa:
#         all_letters.add(char)
# print(len(all_letters))
# all_letters = list(all_letters)
# for letter in all_letters:
#     print(letter)
# print(''.join(all_letters))


# with open('all_letters.txt', 'w') as f:
#     for letter in all_letters:
#         # write each item on a new line
#         f.write("%s\n" % letter)
#     print('Done saving alphabet')




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)


def X2Tensor(ipa):
    # # Using panphon
    # return torch.tensor([[vec] for vec in ft.word_to_vectors(ipa)], dtype=torch.float)

    # Using one-hot encoding
    res = torch.zeros(len(ipa), 1, n_letters) #Create a zeros tensor
    #iterate through all the characters in the name
    for index, letter in enumerate(ipa):
        pos = all_letters.index(letter)
        res[index][0][pos] = 1 #Assign a value for each pos value
    return res


def batched_X2Tensor(words, max_word_size):
    # # Using panphon
    # res = []
    # for ipa in words:
    #     tensor = torch.tensor([vec for vec in ft.word_to_vectors(ipa)], dtype=torch.float)
    #     if len(tensor.shape) == 2:
    #         res.append(tensor)
    # return pad_sequence(res)

    # Using one-hot encoding
    res = torch.zeros(max_word_size, len(words), n_letters)
    for word_index, word in enumerate(words):
        for letter_index, letter in enumerate(word):
            pos = all_letters.index(letter)
            res[letter_index][word_index][pos] = 1
    return res

# out_ = batched_X2Tensor(['pɔl ɑtlɛt', 'd͡ʒɔɹd͡ʒ wɑʃɪŋtən'], 17)
# print(out_)
# print(out_.size())

def y2Tensor(label):
    neClass = 0
    if label == 'PER': neClass = 0
    elif label == 'LOC': neClass = 1
    elif label == 'ORG': neClass = 2
    return torch.tensor([neClass], dtype=torch.float)

def batched_y2Tensor(labels):
    res = []
    for label in labels:
        neClass = 0
        if label == 'PER': neClass = 0
        elif label == 'LOC': neClass = 1
        elif label == 'ORG': neClass = 2
        res.append(neClass)
    return torch.tensor(res, dtype=torch.float)


def dataloader(npoints, X_, y_):
    res = []
    for i in range(npoints):
        index_ = np.random.randint(len(X_))
        ipa, label = X_.iloc[index_], y_.iloc[index_]
        res.append((ipa, label, X2Tensor(ipa), y2Tensor(label)))
    return res

# print(dataloader(2, X_train, y_train))

def batched_dataloader(npoints, X_, y_, verbose=False, device = 'cpu'):
    names = []
    langs = []
    X_lengths = []
    
    i = 0
    while i < npoints:
        index_ = np.random.randint(len(X_))

        # I think this error comes from your [i] notation, which is trying to look for the DataFrame index value of 0, which doesn't exist. Try replacing every instance of [i] with .iloc[i].
        word, tag = X_.iloc[index_], y_.iloc[index_]
        # # Using panphon
        # if len(ft.word_to_vectors(word)) <= 0:
        #     continue
        # X_lengths.append(len(ft.word_to_vectors(word)))

        # Using one-hot encoding
        if len(word) <= 0: continue
        X_lengths.append(len(word))
        
        i += 1
        
        names.append(word)
        langs.append(tag)
    max_length = max(X_lengths)
    
    
    names_rep = batched_X2Tensor(names, max_length).to(device)
    langs_rep = batched_y2Tensor(langs).to(device)
    
    packed_names_rep = pack_padded_sequence(names_rep, X_lengths, enforce_sorted = False)
    
    if verbose:
        print(names_rep.shape, packed_names_rep.data.shape)
        print('--')
    
    if verbose:
        print(names)
        print('--')
    
    if verbose:
        print('Lang Rep', langs_rep.data)
        print('Batch sizes', packed_names_rep.batch_sizes)
    
    return packed_names_rep.to(device), langs_rep

# print(batched_dataloader(2, X_train, y_train, True))


def train(net, opt, criterion, n_points):
    
    opt.zero_grad()
    total_loss = 0
    
    data_ = dataloader(n_points, X_train, y_train)
    
    total_loss = 0
    
    for name, language, name_ohe, lang_rep in data_:

        hidden = net.init_hidden()

        for i in range(name_ohe.size()[0]):
            output, hidden = net(name_ohe[i:i+1], hidden)
            
        loss = criterion(output, lang_rep)
        loss.backward(retain_graph=True)
        
        total_loss += loss
        
    opt.step()       
    return total_loss/n_points

def infer(net, name, device = "cpu"):
    name_ohe = X2Tensor(name).to(device)

    #get the output
    output, hidden = net(name_ohe)

    if type(hidden) is tuple: #for lSTM
        hidden = hidden[0]
    index = torch.argmax(hidden)

    return output

def eval(net, n_points, X_, y_, device = "cpu"):
    "Evaluation function"

    net = net.eval().to(device)
    data_ = dataloader(n_points, X_, y_)
    correct = 0

    #iterate
    for name, language, name_ohe, lang_rep in data_:

        #get the output
        
        output = infer(net, name, device)
        # print(name, lang_rep, output, torch.argmax(output))
        if lang_rep.to(device) == torch.argmax(output):
            correct += 1

    accuracy = correct/n_points
    return accuracy

class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_net, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden = None):
        out, hidden = self.lstm_cell(input, hidden)
        output = self.h2o(hidden[0].view(-1, self.hidden_size))
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self, batch_size = 1):
        return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))


def train_batch(net, opt, criterion, n_points, device = 'cpu'):
    
    net.train().to(device)
    opt.zero_grad()
    
    batch_input, batch_groundtruth = batched_dataloader(n_points, X_train, y_train, False, device)
    batch_groundtruth = batch_groundtruth.type(torch.LongTensor)

    output, hidden = net(batch_input)
    
    loss = criterion(output, batch_groundtruth.to(device))
    
    loss.backward()
    opt.step()
    return loss

def train_setup(net, lr = 0.01, n_batches = 100, batch_size = 10, momentum = 0.9, display_freq=5, device = 'cpu'):
    net = net.to(device)
    criterion = nn.NLLLoss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    loss_arr = np.zeros(n_batches + 1)
    
    for i in range(n_batches):
        loss_arr[i+1] = (loss_arr[i]*i + train_batch(net, opt, criterion, batch_size, device))/(i + 1)
        
        if i%display_freq == 0:
            
            print('Iteration', i, 'Loss', loss_arr[i])
            print('Accuracy:', eval(net, 1000, X_test, y_test, device))
            


# n_letters = 24
n_classes = 3
n_hidden = 128
net = LSTM_net(n_letters, n_hidden, n_classes)
net.to(device)
train_setup(net, lr=0.3, n_batches=8000, batch_size=2048, display_freq=500, device = device)
torch.save(net.state_dict(), './onehot_model.pth')