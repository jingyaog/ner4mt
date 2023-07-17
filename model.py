import torch
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from dotenv import load_dotenv
from datasets import Dataset, load_metric
import tokenize_noargs
import numpy as np
import seqeval


if torch.cuda.is_available():
    print(f"there are {torch.cuda.device_count} GPU(s) available.")
    device = torch.device("cuda")

else:
    print("No GPUs found; using CPU instead.")
    device = torch.device("cpu")


load_dotenv()
key = os.getenv('HUGAUTH')
language = 'ukrainian_annotated'
code = 'ukr-Cyrl'

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_auth_token = key)
tagmap = tokenize_noargs.create_tagmap("tag_map.txt")

dataset = tokenize_noargs.create_dataset(language)
tokenized_dataset = dataset.map(tokenize_noargs.wrapper(code, tokenizer, tagmap), batched = True)
# tokenized_dataset = tokenized_dataset.map(tokenize_noargs.truncate)
tokenized_dataset = tokenized_dataset.train_test_split(test_size = 0.1)
tag_list = list(tagmap.keys())


model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels = len(tag_list), use_auth_token = key)
print(model.config.vocab_size)

BATCH_SIZE = 64
TASK = "ner"

args = TrainingArguments(
    f"test-{TASK}",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=1e-5,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[tagmap[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[tagmap[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], 
            "recall": results["overall_recall"], 
            "f1": results["overall_f1"], 
            "accuracy": results["overall_accuracy"]}

print(len(tokenized_dataset['train']))
print(len(tokenized_dataset['test']))

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model('ukr_ner.model')