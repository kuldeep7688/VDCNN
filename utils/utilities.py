import torch
import torch.nn.functional as F
import numpy as np
import pyprind
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


CSV_FIELDS_DICT = {

}


def tokenizer(text):
    s = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
    return [l for l in list(text.lower()) if l in s]


def print_number_of_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters in the model are : {}".format(params))
    return


def calculate_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    preds, ind= torch.max(F.softmax(preds, dim=-1), 1)
    correct = (ind == y).float()
    acc = correct.sum()/float(len(correct))
    return acc


def binary_accuracy(y, out):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    preds = torch.round(out)
    correct = (preds == y).float()
    acc = correct.sum() / float(len(correct))
    return acc


def roc_auc_score_fixed(y_true, y_pred):
    if len(np.unique(y_true)) == 1:  # bug in roc_auc_score
        return 0.5
    return roc_auc_score(y_true, y_pred)


def get_avg_roc_value(y_fin, output_fin):
    n = len(y_fin)
    out_list = []
    y_list = []
    for i in range(n):
        out_list.extend(list(output_fin[i]))
        y_list.extend(list(y_fin[i]))
    roc = roc_auc_score_fixed(y_list, out_list)

    return roc


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            # if self.y_vars is not None and len():  # we will concatenate y into a single tensor
            #     y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            if self.y_vars:  # we will concatenate y into a single tensor
                y = getattr(batch, self.y_vars)
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


def train_epoch(
    model, iterator, optimizer, criterion, n_classes,
    print_stats_at_step, max_grad_norm
):
    avg_tr_loss = 0.0
    avg_accuracy = 0.0

    predictions_array = []
    true_labels_array = []

    model.train()
    steps = 0
    if n_classes <= 2:
        avg_f1 = 0.0
        tqdm_iterator = tqdm(iterator)
        for x, labels in tqdm_iterator:
            outputs = model(x).squeeze(1)
            predicted_probabilities = torch.sigmoid(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            step_loss = loss.item()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()
            model.zero_grad()

            avg_tr_loss += step_loss
            predictions_array.extend(torch.round(predicted_probabilities).detach().cpu().tolist())
            true_labels_array.extend(labels.detach().cpu().tolist())

            if steps % print_stats_at_step == 0 and steps != 0:
                temp_acc = accuracy_score(true_labels_array, predictions_array)
                temp_f1 = f1_score(true_labels_array, predictions_array)

                avg_accuracy += temp_acc
                avg_f1 += temp_f1
                tqdm_iterator.set_description(
                    f'Iter {steps}| Avg. Tr Loss: {(avg_tr_loss / steps):.3f}| tr_st_acc: {temp_acc:.3f}| tr_st_f1: {temp_f1:.3f}'
                )
                predictions_array = []
                true_labels_array = []

            steps += 1
        return avg_tr_loss / steps , avg_accuracy / ((steps // print_stats_at_step) + 1), avg_f1 / ((steps // print_stats_at_step) + 1)
    elif n_classes > 2:
        tqdm_iterator = tqdm(iterator)
        for x, labels in tqdm_iterator:
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            avg_tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()
            model.zero_grad()

            _, predictions = torch.max(F.softmax(outputs, dim=-1), 1)
            predictions_array.extend(predictions.detach().cpu().tolist())
            true_labels_array.extend(labels.detach().cpu().tolist())

            if steps % print_stats_at_step == 0 and steps != 0:
                temp_acc = accuracy_score(true_labels_array, predictions_array)

                avg_accuracy += temp_acc
                tqdm_iterator.set_description(
                    f'Iter {steps}| Avg. Tr Loss: {(avg_tr_loss / steps):.3f}| tr_st_acc: {temp_acc:.3f}'
                )
                predictions_array = []
                true_labels_array = []

            steps += 1
        return avg_tr_loss / steps , avg_accuracy / ((steps // print_stats_at_step) + 1)


def evaluate_epoch(
    model, iterator, criterion, n_classes,
    print_stats_at_step
):
    avg_eval_loss = 0.0
    avg_accuracy = 0.0

    predictions_array = []
    true_labels_array = []

    model.eval()
    steps = 0
    if n_classes <= 2:
        avg_f1 = 0.0
        with torch.no_grad():
            tqdm_iterator = tqdm(iterator)
            for x, labels in tqdm_iterator:
                outputs = model(x).squeeze(1)
                predicted_probabilities = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                avg_eval_loss += loss.item()

                predictions_array.extend(torch.round(predicted_probabilities).detach().cpu().tolist())
                true_labels_array.extend(labels.detach().cpu().tolist())

                if steps % print_stats_at_step == 0 and steps != 0:
                    temp_acc = accuracy_score(true_labels_array, predictions_array)
                    temp_f1 = f1_score(true_labels_array, predictions_array)

                    avg_accuracy += temp_acc
                    avg_f1 += temp_f1
                    tqdm_iterator.set_description(
                        f'Iter {steps}| Avg. Eval Loss: {(avg_eval_loss / steps):.3f}| ev_st_acc: {temp_acc:.3f}| ev_st_f1: {temp_f1:.3f}'
                    )
                    predictions_array = []
                    true_labels_array = []

                steps += 1
        return avg_eval_loss / steps , avg_accuracy / ((steps // print_stats_at_step) + 1), avg_f1 / ((steps // print_stats_at_step) + 1)
    elif n_classes > 2:
        with torch.no_grad():
            tqdm_iterator = tqdm(iterator)
            for x, labels in tqdm_iterator:
                outputs = model(x)
                loss = criterion(outputs, labels)
                avg_eval_loss += loss.item()

                _, predictions = torch.max(F.softmax(outputs, dim=-1), 1)
                predictions_array.extend(predictions.detach().cpu().tolist())
                true_labels_array.extend(labels.detach().cpu().tolist())

                if steps % print_stats_at_step == 0 and steps != 0:
                    temp_acc = accuracy_score(true_labels_array, predictions_array)

                    avg_accuracy += temp_acc
                    tqdm_iterator.set_description(
                        f'Iter {steps}| Avg. Eval Loss: {(avg_eval_loss / steps):.3f}| tr_eval_acc: {temp_acc:.3f}'
                    )
                    predictions_array = []
                    true_labels_array = []

                steps += 1
        return avg_eval_loss / steps , avg_accuracy / ((steps // print_stats_at_step) + 1)


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
