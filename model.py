import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv


MAX_LENGTH = 128 # trying this out for now


if torch.cuda.is_available():
    print(f"there are {torch.cuda.device_count} GPU(s) available.")
    device = torch.device("cuda")

else:
    print("No GPUs found; using CPU instead.")
    device = torch.device("cpu")

"""
A class to represent our input dataset.
Needs to be decorated to actually adjust to our dataset.
"""
class NER_Dataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__ (self, idx):
        return self.data[idx]
    
load_dotenv()
key = os.getenv('HUGAUTH')

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", use_auth_token = key)
model = BertModel.from_pretrained("bert-base-multilingual-cased", use_auth_token = key)

text = "Este es un texto de prueba."
encoded_input = tokenizer(text, return_tensors = 'pt')
print(encoded_input)
# output = model(**encoded_input)

    


