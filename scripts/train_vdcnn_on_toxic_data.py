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

from model import model
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
    default=1024
)
parser.add_argument(
    "--save_path_for_model", help="Mention the path for saving the model.",
    type=str,
    default="data/models/vdcnn_toxic_model.tar"
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
    default="train_torch.csv"
)
parser.add_argument(
    "--val_file_name", help="Mention the validation csv file name",
    type=str,
    default="test_torch.csv"
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


arguments = parser.parse_args()
N_CLASSES = 6
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


if __name__ == "__main__":
    text_field = data.Field(
        sequential=True,
        use_vocab=True,
        fix_length=CHAR_LENGTH,
        tokenize=tokenizer,
        batch_first=True
    )
    label_field = data.Field(
        sequential=False,
        use_vocab=False,
        is_target=True
    )
    csv_fields = [
        ("id", None),
        ("comment_text", text_field),
        ("toxic", label_field),
        ("severe_toxic", label_field), ("threat", label_field),
        ("obscene", label_field), ("insult", label_field),
        ("identity_hate", label_field)
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


    traindl, valdl = data.BucketIterator.splits(
        datasets=(trainds, valds),
        batch_sizes=(BATCH_SIZE_TRAIN, BATCH_SIZE_VALIDATION),
        sort_key=lambda x: x.comment_text,
        repeat=False,
        device=DEVICE
    )

    del trainds, valds

    save_dict_to_disk(text_field.vocab, VOCAB_PATH)

    train_dl = BatchWrapper(traindl, "comment_text",
                            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    valid_dl = BatchWrapper(valdl, "comment_text",
                            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

    del traindl, valdl

    vocab_size = len(text_field.vocab.stoi)
    model = model.MDCNN(EMBEDDING_DIM, vocab_size, N_CLASSES)
    print(model)
    print_number_of_trainable_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.to(DEVICE)
    criterion.to(DEVICE)

    base_dev_roc = 0.0
    for epoch in range(EPOCHS):

        train_loss, train_acc, train_roc, train_roc_main = train(model, train_dl, optimizer, criterion)
        valid_loss, valid_acc, valid_roc, valid_roc_main = evaluate(model, valid_dl, criterion)
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
        if base_dev_roc < valid_roc_main:
            is_best = True,
            base_dev_roc = valid_roc_main

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': valid_loss,
            'best_dev_accuracy': valid_acc
        }, is_best, MODEL_PATH)
