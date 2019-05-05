# importing required libraries
import os
import sys
import torch
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
from utils.utilities import train, evaluate, print_number_of_trainable_parameters, \
    save_checkpoint, load_check_point, tokenizer, BatchWrapper, save_dict_to_disk

parser = ArgumentParser()
parser.add_argument(
    "--embedding_dim", help="Mention the dimension of embedding.",
    type=int,
    default=16
)
parser.add_argument(
    "--char_length", help="Fix the sentence length for each sentence.",
    type=int,
    default=1014
)
parser.add_argument(
    "--save_path_for_model", help="Mention the path for saving the model.",
    type=str,
    default="data/models/vdcnn_model.tar"
)
parser.add_argument(
    "--save_path_for_vocab", help="Mention the path for saving the model.",
    type=str,
    default="data/models/vdcnn_toxic_dict.pkl"
)
parser.add_argument(
    "--device", help="Mention the device to be used cuda or cpu,",
    type=str,
    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
parser.add_argument(
    "--csv_folder_path", help="Mention the folder path where train, test and validation csv files are stored.",
    type=str,
    default="data/toxic_competition_data/"
)
parser.add_argument(
    "--train_file_name", help="Mention the train csv file name.",
    type=str,
    default="train.csv"
)
parser.add_argument(
    "--val_file_name", help="Mention the validation csv file name",
    type=str,
    default="test.csv"
)
parser.add_argument(
    "--train_batch_size", help="Mention the batch size for training the data.",
    type=int,
    default=64
)
parser.add_argument(
    "--val_batch_size", help="Mention the batch size for validation the data.",
    type=int,
    default=64
)
parser.add_argument(
    "--epochs", help="Mention the number of epochs to train the data on.",
    type=int,
    default=1
)
parser.add_argument(
    "--depth", help="Mention the maximum depth for the model.",
    type=int,
    default=9
)
parser.add_argument(
    "--use_shortcut", help="Mention whether to use shortcuts or not while training.",
    type=bool,
    default=False
)
parser.add_argument(
    "--k_max_k", help="Mention the value of K for K_max pool in just before linear layers.",
    type=int,
    default=8
)
parser.add_argument(
    "--pool_type", help="Mention whether to use k_max or max_pool for pooling operations.",
    type=str,
    default="max_pool"
)
parser.add_argument(
    "--linear_layer_size", help="Mention the size of the linear layer to be used in model.",
    type=int,
    default=2048
)
parser.add_argument(
    "--use_batch_norm_for_linear", help="Mention whether to use batch norm or dropout to regularize linear layers.",
    type=bool,
    default=False
)
parser.add_argument(
    "--linear_dropout", help="Mention the drop out fraction amount if using dropout for regularization in linear layers.",
    type=float,
    default=0.4
)
parser.add_argument(
    "--metric", help="Mention which metric to use for saving the models.",
    type=str,
    default="accuracy"
)
parser.add_argument(
    "--n_classes", help="Mention number of classes in the data",
    type=int,
    default=2
)


arguments = parser.parse_args()
N_CLASSES = arguments.n_classes
EPOCHS = arguments.epochs
EMBEDDING_DIM = arguments.embedding_dim
CHAR_LENGTH = arguments.char_length
MODEL_PATH = arguments.save_path_for_model
VOCAB_PATH = arguments.save_path_for_vocab
DEVICE = arguments.device
CSV_FOLDER_PATH = arguments.csv_folder_path
TRAIN_FILE_NAME = arguments.train_file_name
VALIDATION_FILE_NAME = arguments.val_file_name
BATCH_SIZE_TRAIN = arguments.train_batch_size
BATCH_SIZE_VALIDATION = arguments.val_batch_size
SHORTCUT = arguments.use_shortcut
DEPTH = arguments.depth
POOL_TYPE = arguments.pool_type
K_MAX_K = arguments.k_max_k
LINEAR_LAYER_SIZE = arguments.linear_layer_size
USE_BATCHNORM_FOR_LINEAR = arguments.use_batch_norm_for_linear
LINEAR_DROPOUT = arguments.linear_dropout
METRIC = arguments.metric


if __name__ == "__main__":
    text_field = data.Field(
        sequential=True,
        use_vocab=True,
        fix_length=CHAR_LENGTH,
        tokenize=tokenizer,
        batch_first=True
    )
    if N_CLASSES <= 2:
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
        path=CSV_FOLDER_PATH,
        format="csv",
        train=TRAIN_FILE_NAME,
        validation=VALIDATION_FILE_NAME,
        fields=csv_fields,
        skip_header=True
    )
    text_field.build_vocab(trainds)
    label_field.build_vocab(trainds)

    train_dl, valid_dl = data.BucketIterator.splits(
        datasets=(trainds, valds),
        batch_sizes=(BATCH_SIZE_TRAIN, BATCH_SIZE_VALIDATION),
        sort_key=lambda x: x.text,
        repeat=False,
        device=DEVICE
    )
    del trainds, valds
    save_dict_to_disk(text_field.vocab, VOCAB_PATH)
    # train_dl = BatchWrapper(traindl, "text", "label")
    # valid_dl = BatchWrapper(valdl, "text", "label")
    # del traindl, valdl
    vocab_size = len(text_field.vocab.stoi)
    if N_CLASSES <= 2:
        model = get_vdcnn(
            DEPTH, EMBEDDING_DIM, vocab_size, N_CLASSES - 1,
            shortcut=SHORTCUT, pool_type=POOL_TYPE,
            final_k_max_k=K_MAX_K, linear_layer_size=LINEAR_LAYER_SIZE,
            use_batch_norm_for_linear=USE_BATCHNORM_FOR_LINEAR,
            linear_dropout=LINEAR_DROPOUT
        )
        criterion = nn.BCEWithLogitsLoss()
    else:
        model = get_vdcnn(
            DEPTH, EMBEDDING_DIM, vocab_size, N_CLASSES,
            shortcut=SHORTCUT, pool_type=POOL_TYPE,
            final_k_max_k=K_MAX_K, linear_layer_size=LINEAR_LAYER_SIZE,
            use_batch_norm_for_linear=USE_BATCHNORM_FOR_LINEAR,
            linear_dropout=LINEAR_DROPOUT
        )
        criterion = nn.CrossEntropyLoss()
    print(model)
    print_number_of_trainable_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.to(DEVICE)
    criterion.to(DEVICE)
    base_dev_metric = 0.0
    for epoch in range(EPOCHS):
        if N_CLASSES <= 2:
            train_loss, train_acc, train_roc, train_roc_main = train(model, train_dl, optimizer, criterion, N_CLASSES)
            valid_loss, valid_acc, valid_roc, valid_roc_main = evaluate(model, valid_dl, criterion, N_CLASSES)
            print(
                f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} |\
                 Train ROC: {train_roc * 100:.2f} | Train Acc: {train_acc * 100:.2f}%'
            )
            print(
                f'| Epoch: {epoch + 1:02} | Val. Loss: {valid_loss:.3f} | \
                Val. ROC: {valid_roc * 100:.2f} | Val. Acc: {valid_acc * 100:.2f}% |'
            )
            print()
            print(f'| Train Main ROC: {train_roc_main * 100:.2f} | Val. Main ROC: {valid_roc_main * 100:.2f} ')
            is_best = False
            if METRIC == "accuracy":
                if base_dev_metric < valid_acc:
                    is_best = True,
                    base_dev_metric = valid_acc
            else:
                if base_dev_metric < valid_roc_main:
                    is_best = True,
                    base_dev_metric = valid_roc_main
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': valid_loss,
                'best_dev_accuracy': valid_acc
            }, is_best, MODEL_PATH)
        else:
            train_loss, train_acc = train(model, train_dl, optimizer, criterion, N_CLASSES)
            valid_loss, valid_acc = evaluate(model, valid_dl, criterion, N_CLASSES)
            print(f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'| Epoch: {epoch + 1:02} | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')
            print()
            is_best = False
            if base_dev_metric < valid_acc:
                is_best = True,
                base_dev_roc = valid_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': valid_loss,
                'best_dev_accuracy': valid_acc
            }, is_best, MODEL_PATH)
    print("DONE .......................... :D")
