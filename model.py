import torch
from datasets import NER_Dataset
import os
from transformers import BertTokenizer, BertModel
from transformers import Trainer, TrainingArguments
from dotenv import load_dotenv
import seqeval


if torch.cuda.is_available():
    print(f"there are {torch.cuda.device_count} GPU(s) available.")
    device = torch.device("cuda")

else:
    print("No GPUs found; using CPU instead.")
    device = torch.device("cpu")


load_dotenv()
key = os.getenv('HUGAUTH')

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", use_auth_token = key)
model = BertModel.from_pretrained("bert-base-multilingual-cased", use_auth_token = key)

dataset = NER_Dataset(tokenizer, "tag_map.txt", "ukr-Cyrl")

TRAIN_SIZE = 0.8
EPOCHS = 10
BATCH_SIZE = 64

train_len = int(len(dataset)*TRAIN_SIZE)
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
logging_steps = len(train_set) // BATCH_SIZE

training_args = TrainingArguments(
output_dir="results",
num_train_epochs=EPOCHS,
per_device_train_batch_size=BATCH_SIZE,
per_device_eval_batch_size=BATCH_SIZE,
evaluation_strategy="epoch",
disable_tqdm=False,
logging_steps=logging_steps)

trainer = Trainer(model = model,
                  args = training_args,
                  train_dataset = train_set,
                  eval_dataset = val_set,
                  data_collator=dataset.collate_tokenize)

trainer.train()


