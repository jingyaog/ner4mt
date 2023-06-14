import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import numpy as np
from transformers import get_linear_schedule_with_warmup
from dotenv import load_dotenv
from math import random
import time
import datetime

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

training_data = NER_Dataset("")
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    


load_dotenv()
key = os.getenv('HUGAUTH')

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", use_auth_token = key)
model = BertModel.from_pretrained("bert-base-multilingual-cased", use_auth_token = key)
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
epochs = 4
total_steps = len(train_dataloader)*epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
# output = model(**encoded_input)


def pre_processing(sentences):
    input_ids = []
    attention_masks = []
    for s in sentences:
        encoded_dict = tokenizer.encode_plus(
            s,
            add_special_tokens = True,
            max_length = MAX_LENGTH,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def train():
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    model.train()
    for epoch in range(0, epochs):
        print("")
        print(f"'========= EPOCH {epoch}/{epochs} =========='")
        print("training.....")
        total_train_loss = 0
        t0 = time.time()
        for step, batch in enumerate(train_dataloader):

            if(step % 40 == 0 and step != 0):
                elapsed = format_time(time.time() - t0)
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            

            model.zero_grad()

            loss, logits = model(b_input_ids,
                                 token_type_ids = None,
                                 attention_mask = b_input_mask,
                                 labels = b_labels)
            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
            
            optimizer.step()
            scheduler.step()



