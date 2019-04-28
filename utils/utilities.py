import torch
import numpy as np
import pyprind
import pickle
from sklearn.metrics import roc_auc_score


def tokenizer(text):
    s = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
    return [l for l in list(text.lower()) if l in s]


def print_number_of_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters in the model are : {}".format(params))
    return


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    preds = torch.round(torch.sigmoid(preds))
    correct = (preds == y).float()
    acc = correct.sum() / float(len(correct))
    return acc / 6


def roc_auc_score_fixed(y_true, y_pred):
    if len(np.unique(y_true)) == 1:  # bug in roc_auc_score
        return 0.5
    return roc_auc_score(y_true, y_pred)


def get_avg_roc_value(y, output):
    out = torch.sigmoid(output)
    out = out.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    # dividing the predictions according to the siz classes
    roc_list = []
    for i in range(6):
        roc = roc_auc_score_fixed(y[:, i], out[:, i])
        roc_list.append(roc)

    # average
    return sum(roc_list) / float(6)


def get_avg_roc_value_2(y_fin, output_fin):
    n = len(y_fin)
    out_list = [[], [], [], [], [], []]
    y_list = [[], [], [], [], [], []]
    for i in range(n):
        for j in range(6):
            out_list[j].extend(list(output_fin[i][:, j]))
            y_list[j].extend(list(y_fin[i][:, j]))

    roc_list = []
    for i in range(6):
        roc_list.append(roc_auc_score_fixed(y_list[i], out_list[i]))

    return sum(roc_list) / 6


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is not None:  # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_roc = 0
    all_y = []
    all_out_list = []

    model.train()
    bar = pyprind.ProgBar(len(iterator), bar_char='█')
    for x, y in iterator:
        optimizer.zero_grad()
        outputs = model(x).squeeze(1)
        loss = criterion(outputs, y)
        acc = binary_accuracy(outputs, y)
        roc = get_avg_roc_value(y, outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_roc += roc

        all_out_list.append(torch.sigmoid(outputs).cpu().detach().numpy())
        all_y.append(y.cpu().detach().numpy())

        bar.update()
    roc_main = get_avg_roc_value_2(all_y, all_out_list)
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_roc / len(iterator), roc_main


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_roc = 0

    all_y = []
    all_out_list = []
    model.eval()

    with torch.no_grad():
        bar = pyprind.ProgBar(len(iterator), bar_char='█')
        for x, y in iterator:
            outputs = model(x).squeeze(1)
            loss = criterion(outputs, y)
            acc = binary_accuracy(outputs, y)
            roc = get_avg_roc_value(y, outputs)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_roc += roc

            all_out_list.append(torch.sigmoid(outputs).cpu().detach().numpy())
            all_y.append(y.cpu().detach().numpy())

            bar.update()
    roc_main = get_avg_roc_value_2(all_y, all_out_list)
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_roc / len(iterator), roc_main


def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation loss did not improve")
    return


def load_check_point(model, model_path):
    resume_weights = model_path
    checkpoint = torch.load(resume_weights)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_dev_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("Best Dev Accuracy is {}".format(best_accuracy))
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))
    return model

def save_dict_to_disk(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load_dict_from_disk(path):
    with open(path, 'rb') as fp:
        d = pickle.load(fp)
    return d