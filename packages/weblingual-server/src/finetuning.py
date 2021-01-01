import sys
import json
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.autograd import Variable
from torchsummary import summary

model_loc = sys.argv[1]
destination_loc = sys.argv[2]
dataset_loc = sys.argv[3]

if not (model_loc or destination_loc or dataset_loc):
    print("Usage: python finetuning.py <model_location> <destination_location> <dataset_location>")
    sys.exit(1)

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")

MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 5

label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]

import re

def decontracted(phrase, nextw):
    # specific
    if phrase == "wo" and nextw == "n't":
        return "will"
    elif phrase == "ca" and nextw == "n't":
        return "can"
    else:
        # general
        phrase = re.sub(r"n\'t", "not", phrase)
        phrase = re.sub(r"\'re", "are", phrase)
        phrase = re.sub(r"\'s", "is", phrase)
        phrase = re.sub(r"\'d", "would", phrase)
        phrase = re.sub(r"\'ll", "will", phrase)
        phrase = re.sub(r"\'t", "not", phrase)
        phrase = re.sub(r"\'ve", "have", phrase)
        phrase = re.sub(r"\'m", "am", phrase)
        return phrase

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def get_labels(initial, len):
    return [initial] + [(initial + (initial % 2)) for i in range(len - 1)]

def load_btc_dataset(location):
    '''
    Loads the broad twitter corpus dataset by Leon Derczynski, Kalina Bontcheva, and Ian Roberts. Proceedings of COLING, pages 1169-1179 2016.
    Returns an object combining all data from section A to H
    Something like [[['hello', 'world'], [0, 0]], [['hello', 'Paris'], [0, 8]]], a TOTAL_DATA * MAX_LENGTH * 2 tensor.
    '''
    files = ['a', 'b', 'h', 'g', 'e', 'f']
    ttokens = []
    llabels = []
    for name in files:
        file_name = location + "/" + name + ".json" # The file name of the dataset
        file = open(file_name, 'r')
        lines = file.readlines()
        for line in lines: # This is one of the entries in the dataset
            json_obj = json.loads(line.strip())
            tokens = [101]
            labels = [0]

            for idx, t in enumerate(json_obj['tokens']):
                idn = idx + 1 if idx < len(json_obj['tokens']) - 1 else idx
                real_token = decontracted(t, json_obj['tokens'][idn])
                ids = tokenizer(t)['input_ids'][1:-1]
                if 0 == len(ids):
                    pass
                else:
                    tokens.extend([ids[j] for j in range(len(ids))])
                    labels.extend(get_labels(label_list.index(json_obj["entities"][idx]), len(ids)))

            tokens = tokens + [102]
            labels = labels + [0]

            if len(tokens) > MAX_LENGTH:
                tokens = tokens[:MAX_LENGTH - 1] + [102]
                labels = labels[:MAX_LENGTH - 1] + [0]
            else:
                tokens = tokens + [0 for i in range(MAX_LENGTH - len(tokens))]
                labels = labels + [0 for i in range(MAX_LENGTH - len(labels))]

            ttokens.append(tokens)
            llabels.append(labels)

    for line in range(len(ttokens)):
        print(tokenizer.decode(ttokens[line]), llabels[line])

    return TensorDataset(torch.tensor(ttokens), torch.tensor(llabels))

dataset = load_btc_dataset(dataset_loc)

print(dataset)

ds_val, ds_train = random_split(dataset, [100, len(dataset)-100])
loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE)
loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE)
model = AutoModelForTokenClassification.from_pretrained(model_loc).to('cuda')
model.train()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)
criterion = nn.BCEWithLogitsLoss()

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def evaluate(net, loss, dataloader):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for tokens, labels in dataloader:
            tokens, labels = tokens.cuda(), to_one_hot(labels, 9).cuda()

            out = model(tokens).logits

            for idx, j in enumerate(tokens):
                token_len = np.where(j.cpu().numpy() == 102)[0][0]
                mean_loss += loss(out[idx][:token_len], labels[idx][:token_len]).float()
                mean_acc += get_accuracy_from_logits(out[idx][:token_len], labels[idx][:token_len])

            count += 1

    return mean_acc / (count * BATCH_SIZE), mean_loss / count



def train(model, opti, loss, loader):
    for epoch in range(EPOCHS):
        for i, (tokens, labels) in enumerate(loader):
            if (i+1) % 50 == 0:
                print("Iteration done for {}".format(i))
            opti.zero_grad()
            loss_val = torch.tensor(0.0).cuda()
            #[tokens, labels] = i
            tokens, labels = tokens.cuda(), to_one_hot(labels, n_dims=9).cuda()
            out = model(tokens).logits
            for idx, j in enumerate(tokens):
                token_len = np.where(j.cpu().numpy() == 102)[0][0]
                loss_val += loss(out[idx][:token_len], labels[idx][:token_len])
            #loss_val = loss(out, labels)
            loss_val.cuda()
            loss_val.backward()
            opti.step()

        
        print(evaluate(model, loss, loader_val))
'''
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

for param in model.bert.parameters():
    param.requires_grad = False


train(model, optimizer, criterion, loader_train)
'''
#for name, param in model.bert.named_parameters():
#    param.requires_grad = False

for param in model.parameters():
    param.requires_grad = True


train(model, optimizer, criterion, loader_train)

model.save_pretrained(destination_loc)
print("Fine tuning complete.")