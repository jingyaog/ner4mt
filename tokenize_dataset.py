import os
import pandas as pd
import epitran
import panphon
from transformers import AutoTokenizer
from datasets import Dataset


def process_file(df, filepath):
    sentences = []
    labelings = []
    with open(filepath, "r", encoding = "utf-8") as file:
        sentence = []
        labeling = []
        for line in file:
            if(line == "\n"):
                df.loc[len(df)] = {"Tokens": sentence, "Tags": labeling}
                sentence = []
                labeling = []
            else:
                word, tag = line.split()
                sentence.append(word)
                labeling.append(tag)


def create_dataset(language):
    df = pd.DataFrame(columns = ["Tokens", "Tags"])
    if('data' not in os.listdir()):
        print("Data not downloaded!\n")
        raise FileNotFoundError
    for _,_,files in os.walk('data/{}'.format(language)):
        for file in files:
            process_file(df, 'data/{}/{}'.format(language, file))
    return Dataset.from_pandas(df)

def create_tagmap(filepath):
    result = dict()
    idx = 0
    with open(filepath) as file:
        for line in file:
            result[line[:-1]] = idx #to remove the "\n" character
            idx += 1
    return result

def bit_list_to_integer(bit_list):
    res = 0
    for bit in bit_list:
        res = (res << 1) | (1 if bit > 0 else 0)
    return res


def tokenize(tokenizer, code, sentences):
    res = {"input_ids": [], "word_ids": [], "attention_masks": []}
    epi = epitran.Epitran(code)
    ft = panphon.FeatureTable()
    cls, sep = tokenizer("")['input_ids']
    for sentence in sentences:
        input_ids = []
        word_ids = []
        curr_word = 0
        for word in sentence:
            ipa = epi.transliterate(word)
            features = ft.word_to_vector_list(ipa, numeric = True)
            for feature in features:
                input_ids.append(bit_list_to_integer(feature))
                word_ids.append(curr_word)
            curr_word += 1
        input_ids.insert(0, cls)
        input_ids.append(sep)
        word_ids.insert(0, None)
        word_ids.append(None)
        res["input_ids"].append(input_ids)
        res["word_ids"].append(word_ids)
        res["attention_masks"].append([1]*len(word_ids))
    return res

def wrapper(code, tokenizer):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenize(tokenizer, code, list(examples["Tokens"]))
        labels = []
        for i, label in enumerate(examples["Tags"]):
            word_ids = tokenized_inputs["word_ids"][i]
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(tagmap[label[word_idx]])
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize_and_align_labels

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
ukr_dataset = create_dataset("ukrainian")
tagmap = create_tagmap("tag_map.txt")
print(ukr_dataset[0])
ukr_tokenized_dataset = ukr_dataset.map(wrapper("ukr-Cyrl", tokenizer), batched = True)

ukr_tokenized_dataset.to_csv("tester.csv")