directory = "../../lorelei_ukrainian_repr_lang_pack/data/monolingual_text/annotated/"
import argparse
import torch
import torch.nn.functional as F
from src.model import CharacterLevelCNN
from src import utils
import epitran
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()


def predict(args):
    model = CharacterLevelCNN(args, args.number_of_classes)
    state = torch.load(args.model)
    model.load_state_dict(state)
    model.eval()

    processed_input = utils.preprocess_input(args)
    processed_input = torch.tensor(processed_input)
    processed_input = processed_input.unsqueeze(0)
    if use_cuda:
        processed_input = processed_input.to("cuda")
        model = model.to("cuda")
    prediction = model(processed_input)
    probabilities = F.softmax(prediction, dim=1)
    probabilities = probabilities.detach().cpu().numpy()
    return probabilities

def toLabel(x):
    if x == 0: return "PER"
    if x == 1: return "LOC"
    if x == 2: return "ORG"

if __name__ == "__main__":

    epi = epitran.Epitran('ukr-Cyrl')

    all_categories = ["PER", "LOC", "ORG"]
    n_categories = len(all_categories)

    orths = []
    X = []
    true_tags = []
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        with open(file, "r") as f:
            for row in csv.reader(f, delimiter='\t'):
                if len(row) < 2 or row[1] == 'O' or row[1][2:] == 'TTL': continue
                orth, tag = row[0], row[1]
                if epi.transliterate(orth).isdigit(): continue
                if tag[0] == 'B':
                    orths.append(orth)
                    if tag[2:] == "GPE":
                        true_tags.append("LOC")
                    else:
                        true_tags.append(tag[2:])
                if tag[0] == 'I':
                    orths[-1] += ' ' + orth

    for orth in orths:
        X.append(epi.transliterate(orth))


    parser = argparse.ArgumentParser(
        "Testing a pretrained Character Based CNN for text classification"
    )
    parser.add_argument("--model", type=str, help="path for pre-trained model")
    parser.add_argument("--text", type=str, default="I love pizza!", help="text string")
    parser.add_argument("--steps", nargs="+", default=["lower"])

    # arguments needed for the predicition
    parser.add_argument(
        "--alphabet",
        type=str,
        default="ʊð[pu\"?$~#k|>tθɛv&wɪsf-̩d(*labzɑnʒ ,]ʃm.e}:^!/\=ɹəoɡ)+@_ j͡hɔi{ŋʌ`æ;<",
    )
    parser.add_argument("--number_of_characters", type=int, default=69)
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--number_of_classes", type=int, default=3)

    args = parser.parse_args()
    pred_tags = []
    print(len(true_tags))
    i=0
    for ipa in X:
        i+=1
        args.text = ipa
        prediction = predict(args)
        pred_label = max(enumerate(prediction[0]), key=lambda x: x[1])[0]
        pred_tags.append(toLabel(pred_label))
        if i%100 == 0: print(i)

        # print("input : {}".format(args.text))
        # print("prediction : {}".format(prediction))
    
    print(true_tags, pred_tags)

    print("Accuracy: %.3f" % accuracy_score(true_tags, pred_tags))
    print("Micro-avg Precision: %.3f" % precision_score(true_tags, pred_tags, average='micro'))
    print("Macro-avg Precision: %.3f" % precision_score(true_tags, pred_tags, average='macro'))
    print("Micro-avg Recall: %.3f" % recall_score(true_tags, pred_tags, average='micro'))
    print("Macro-avg Recall: %.3f" % recall_score(true_tags, pred_tags, average='macro'))
    print("Micro-avg F1: %.3f" % f1_score(true_tags, pred_tags, average='micro'))
    print("Macro-avg F1: %.3f" % f1_score(true_tags, pred_tags, average='macro'))