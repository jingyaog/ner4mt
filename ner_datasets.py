import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import epitran.vector 
import os


class NER_Dataset(Dataset):
    def process_file(self, filepath):
        sentences = []
        labelings = []
        with open(filepath, "r", encoding = "utf-8") as file:
            sentence = []
            labeling = []
            for line in file:
                if(line == "\n"):
                    self.data.loc[len(self.data)] = {"Tokens": sentence, "Tags": labeling}
                    sentence = []
                    labeling = []
                else:
                    word, tag = line.split()
                    sentence.append(word)
                    labeling.append(self.tag_map[tag])
        return
    
    def get_tags(self, tag_map):
        result = dict()
        idx = 0
        with open(tag_map) as file:
            for line in file:
                result[line[:-1]] = idx #to remove the "\n" character
                idx += 1
        return result

    def __init__(self, tokenizer, tag_map, ipa): # add tokenizer later
        self.tag_map = self.get_tags(tag_map)
        self.tokenizer = tokenizer
        self.cls, self.sep = tokenizer("")['input_ids']
        print("HERE\n")
        print(self.tag_map)
        self.data = pd.DataFrame(columns = ["Tokens", "Tags"])
        self.vwis = epitran.vector.VectorsWithIPASpace('{}'.format(ipa), ['{}'.format(ipa)])
        if('data' not in os.listdir()):
            print("Data not downloaded!\n")
            raise FileNotFoundError
        for _,_,files in os.walk('data'):
            for file in files:
                self.process_file('data/{}'.format(file))
    
    def bit_list_to_integer(self, bit_list):
        res = 0
        for bit in bit_list:
            res = (res << 1) | abs(bit)
        return res
    
    def tokenize(self, tokens, text_labels):
        input_ids = []
        labels = []
        for i in range(len(tokens)):
            #call vectorize on each word
            curr = self.vwis.word_to_segs(tokens[i])
            input_ids.append(self.bit_list_to_integer(curr[0][5]))
            labels.append(text_labels[i])
            #label first, set rest as -100
            for j in range(1, len(curr)):
                input_ids.append(self.bit_list_to_integer(curr[i][5]))
                labels.append(-100)
        # insert sep and cls
        input_ids.insert(self.cls, 0)
        labels.insert(-100, 0)
        input_ids.append(self.sep)
        labels.append(-100)
        attn_mask = [1]*len(labels)
        return {"input_ids": torch.Tensor(input_ids), 
                "labels": torch.Tensor(labels), 
                "attention_masks": attn_mask}


    
    def collate_tokenize(self, batch):
        input_ids = []
        labels = []
        attention_masks = []

        #tokenize
        for item in batch:
            tokenized_dict = self.tokenize(item['Tokens'], item['Tags'])
            input_ids.append(tokenized_dict['input_ids'])
            labels.append(tokenized_dict['labels'])
            attention_masks.append(tokenized_dict['attention_masks'])
        
        #pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True)
        labels = pad_sequence(labels, pad_first = True, padding_value = -100) #double check pad value
        attention_masks = pad_sequence(attention_masks, pad_first = True)

        return input_ids, labels, attention_masks

    def __len__(self):
        return self.data.shape[0]

    def __getitem__ (self, idx):
        tokens, tags = self.data.iloc[idx]
        return {"Tokens": tokens, "Tags": tags}


### testing
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
test = NER_Dataset(tokenizer, "tag_map.txt", 'ukr-Cyrl')
dataloader = DataLoader(test, batch_size = 2, shuffle = True, collate_fn = test.collate_tokenize)
print(test.data.head)
print(test[0])
print(len(test))
print(next(iter(dataloader)))


