# importing required libraries
import os
import sys
import torch
import fire
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# importing local modules
script_path = os.path.abspath('')
sys.path.insert(0, os.path.abspath(script_path))

from model.model import *
from utils.utilities import train_epoch, evaluate_epoch, print_number_of_trainable_parameters, \
    save_checkpoint, load_check_point, tokenizer, BatchWrapper, save_dict_to_disk


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(
    csv_folder_path, save_path_for_model, save_path_for_vocab,
    n_classes=2, epochs=1, embedding_dim=16, char_length=1024,
    train_file_name="train.csv", val_file_name="test.csv",
    train_batch_size=128, val_batch_size=64, use_shortcut=False, depth=9, pool_type="max_pool", k_max_k=8,
    linear_layer_size=2048, use_batch_norm_for_linear=False, linear_dropout=0.4, metric="accuracy",
    device=DEVICE, print_stats_at_step=50, max_grad_norm=1.0
):
    """
    Main loop for training and evaluating on test
    Args:
        n_classes: Mention number of classes in the data
        epochs: Mention the number of epochs to train the data on.
        embedding_dim: Mention the dimension of embedding.
        char_length: Fix the sentence length for each sentence.
        save_path_for_model: Mention the path for saving the model
        save_path_for_vocab: Mention the path for saving the model related files.
        device: Mention the device to be used cuda or cpu,
        csv_folder_path: Mention the folder path where train, test and validation csv files are stored.
        train_file_name: Mention the train csv file name.
        val_file_name: Mention the validation csv file name
        train_batch_size: Mention the batch size for training the data.
        val_batch_size: Mention the batch size for validation the data.
        use_shortcut: Mention whether to use shortcuts or not while training.
        depth: Mention the maximum depth for the model.
        pool_type: Mention whether to use k_max or max_pool for pooling operations.
        k_max_k: Mention the value of K for K_max pool in just before linear layers.
        linear_layer_size: Mention the size of the linear layer to be used in model.
        use_batch_norm_for_linear: Mention whether to use batch norm or dropout to regularize linear layers.
        linear_dropout: Mention the drop out fraction amount if using dropout for regularization in linear layers.
        metric: Mention which metric to use for saving the models.
        print_stats_at_step: number of steps to update display statistics
        max_grad_norm: max grad norm
    Returns:
        type: description
    """
    text_field = data.Field(
        sequential=True,
        use_vocab=True,
        fix_length=char_length,
        tokenize=tokenizer,
        batch_first=True
    )
    if n_classes <= 2:
        label_field = data.Field(
            sequential=False,
            use_vocab=False,
            is_target=True,
            dtype=torch.float
        )
    else:
        label_field = data.Field(
            sequential=False,
            use_vocab=False,
            is_target=True,
        )
    csv_fields = [
        ("label", label_field),
        ("text", text_field),
    ]
    trainds, valds = data.TabularDataset.splits(
        path=csv_folder_path,
        format="csv",
        train=train_file_name,
        validation=val_file_name,
        fields=csv_fields,
        skip_header=True
    )
    text_field.build_vocab(trainds)
    label_field.build_vocab(trainds)

    train_dl, valid_dl = data.BucketIterator.splits(
        datasets=(trainds, valds),
        batch_sizes=(train_batch_size, val_batch_size),
        sort_key=lambda x: x.text,
        repeat=False,
        device=device
    )
    del trainds, valds
    save_dict_to_disk(text_field.vocab, save_path_for_vocab)
    # train_dl = BatchWrapper(traindl, "text", "label")
    # valid_dl = BatchWrapper(valdl, "text", "label")
    # del traindl, valdl
    vocab_size = len(text_field.vocab.stoi)
    if n_classes <= 2:
        model = get_vdcnn(
            depth, embedding_dim, vocab_size, n_classes - 1,
            shortcut=use_shortcut, pool_type=pool_type,
            final_k_max_k=k_max_k, linear_layer_size=linear_layer_size,
            use_batch_norm_for_linear=use_batch_norm_for_linear,
            linear_dropout=linear_dropout
        )
        criterion = nn.BCEWithLogitsLoss()
    else:
        model = get_vdcnn(
            depth, embedding_dim, vocab_size, n_classes,
            shortcut=use_shortcut, pool_type=pool_type,
            final_k_max_k=k_max_k, linear_layer_size=linear_layer_size,
            use_batch_norm_for_linear=use_batch_norm_for_linear,
            linear_dropout=linear_dropout
        )
        criterion = nn.CrossEntropyLoss()
    print(model)
    print_number_of_trainable_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.to(device)
    criterion.to(device)
    base_dev_metric = 0.0
    for epoch in range(epochs):
        if n_classes <= 2:
            train_loss, train_acc, train_f1 = train_epoch(
                model, train_dl, optimizer, criterion, n_classes,
                print_stats_at_step, max_grad_norm
            )
            valid_loss, valid_acc, valid_f1 = evaluate_epoch(
                model, valid_dl, criterion, n_classes,
                print_stats_at_step
            )
            print(
                f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train F1: {train_f1 * 100:.2f} | Train Acc: {train_acc * 100:.2f}% |'
            )
            print(
                f'| Epoch: {epoch + 1:02} | Val. Loss: {valid_loss:.3f} | Val. F1: {valid_f1 * 100:.2f} | Val. Acc: {valid_acc * 100:.2f}% |'
            )
            is_best = False
            if metric == "accuracy":
                if base_dev_metric < valid_acc:
                    is_best = True,
                    base_dev_metric = valid_acc
            else:
                if base_dev_metric < valid_f1:
                    is_best = True,
                    base_dev_metric = valid_f1
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': valid_loss,
                'best_dev_accuracy': valid_acc
            }, is_best, save_path_for_model)
        else:

            train_loss, train_acc = train_epoch(
                model, train_dl, optimizer, criterion, n_classes,
                print_stats_at_step, max_grad_norm
            )
            valid_loss, valid_acc = evaluate_epoch(
                model, valid_dl, criterion, n_classes,
                print_stats_at_step
            )
            print(f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'| Epoch: {epoch + 1:02} | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')
            is_best = False
            if base_dev_metric < valid_acc:
                is_best = True,
                base_dev_roc = valid_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': valid_loss,
                'best_dev_accuracy': valid_acc
            }, is_best, save_path_for_model)

    # best model result of test
    model = load_check_point(model, save_path_for_model)

    if n_classes <= 2:
        test_loss, test_acc, test_f1 = evaluate_epoch(
            model, valid_dl, criterion, n_classes,
            print_stats_at_step
        )
        print(
            f'| TEST RESULT | Test. Loss: {test_loss:.3f} | Test. F1: {test_f1 * 100:.2f} | Test. Acc: {test_acc * 100:.2f}% |'
        )
    else:
        test_loss, test_acc = evaluate_epoch(
            model, valid_dl, criterion, n_classes,
            print_stats_at_step
        )
        print(f'| TEST RESULT | Test. Loss: {test_loss:.3f} | Test. Acc: {test_acc * 100:.2f}% |')

    print("DONE .......................... :D")


if __name__ == "__main__":
    fire.Fire(main)
