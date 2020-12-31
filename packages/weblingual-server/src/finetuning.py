import sys
import json
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

model_loc = sys.argv[1]
destination_loc = sys.argv[2]
dataset_loc = sys.argv[3]

if not (model_loc or destination_loc or dataset_loc):
    print("Usage: python finetuning.py <model_location> <destination_location> <dataset_location>")
    sys.exit(1)

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")

MAX_LENGTH = 512
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

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


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
            tokens = [101] + [tokenizer(t)['input_ids'][1] for t in json_obj["tokens"]] + [102]
            labels = [0] + [label_list.index(l) for l in json_obj["entities"]] + [0]
            if len(tokens) > MAX_LENGTH:
                tokens = tokens[:MAX_LENGTH - 1] + [102]
                labels = labels[:MAX_LENGTH - 1] + [0]
            else:
                tokens = tokens + [0 for i in range(MAX_LENGTH - len(tokens))]
                labels = labels + [0 for i in range(MAX_LENGTH - len(labels))]
            ttokens.append(tokens)
            llabels.append(labels)
    random.shuffle(ttokens)
    random.shuffle(llabels)
    return TensorDataset(torch.tensor(ttokens), torch.tensor(llabels))

dataset = load_btc_dataset(dataset_loc)
loader_train = DataLoader(dataset[1000:], batch_size=32)
loader_eval = DataLoader(dataset[:1000], batch_size=32)
model = AutoModelForTokenClassification.from_pretrained(model_loc).to('cuda')

optimizer = optim.Adam(model.parameters(), lr = 2e-5)
criterion = nn.BCEWithLogitsLoss()

def evaluate(net, criterion, dataloader):
    def get_accuracy_from_logits(logits, labels):
        probs = torch.sigmoid(logits.unsqueeze(-1))
        soft_probs = (probs > 0.5).long()
        acc = (soft_probs.squeeze() == labels).float().mean()
        return acc

    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for tokens, labels in dataloader:
            tokens, labels = tokens.cuda(), to_one_hot(labels).cuda()

            out = model(tokens).logits
            mean_loss += criterion(out, labels).item()
            mean_acc += get_accuracy_from_logits(out, labels)
            count += 1

    return mean_acc / count, mean_loss / count

def train(model, opti, loss, loader):
    for epoch in range(EPOCHS):
        for tokens, labels in loader:
            tokens, labels = tokens.cuda(), to_one_hot(labels).cuda()
            out = model(tokens).logits
            loss = crit(out, labels)
            loss.backward()
            opti.step()
        
        print(evaluate(model, loss, loader_eval))

train(model, optimizer, criterion, loader_train)