import math
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, shortcut=False, pool_type="max_pool"):
        super().__init__()
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.padding = padding
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=self.padding)
        self.batch_norm_1 = nn.BatchNorm1d(out_channels)
        self.conv_2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=self.padding)
        self.batch_norm_2 = nn.BatchNorm1d(out_channels)
        if shortcut:
            self.conv_res = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
            self.batch_norm_res = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.batch_norm_1(out))
        out = self.conv_2(out)
        out = F.relu(self.batch_norm_2(out))

        # downsampled
        if self.pool_type == "k_max":
            k = math.ceil(out.shape[2] / 2.0)
            out = downsample_k_max_pool(out, k=k, dim=2)[0]
        else:
            out = downsample_max_pool(out, 3, 2)

        if self.shortcut is False:
            res_out = self.conv_res(x)
            res_out = F.relu(self.batch_norm_res(res_out))
            out = out + res_out
        return out


class ConvolutionalIdentityBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=1, shortcut=False):
        super().__init__()
        self.shortcut = shortcut
        self.padding = padding
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, padding=self.padding)
        self.batch_norm_1 = nn.BatchNorm1d(in_channels)
        self.conv_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, padding=self.padding)
        self.batch_norm_2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(self.batch_norm_1(out))
        out = self.conv_2(out)
        out = F.relu(self.batch_norm_2(out))
        if self.shortcut:
            out = out + x
        return out


def downsample_max_pool(x, kernel_size, stride):
    pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=1)
    return pool(x)


def downsample_k_max_pool(inp, k, dim):
    return inp.topk(k, dim)


class VDCNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
        self.conv_64 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm_conv_64 = nn.BatchNorm1d(64)
        self.id_64 = ConvolutionalIdentityBlock(64, kernel_size=3, padding=1, shortcut=True)
        self.res_128 = ConvolutionalBlockRes(in_channels=64, out_channels=128, kernel_size=3, padding=1, shortcut=True,
                                             pool_type="k_max")
        self.id_128 = ConvolutionalIdentityBlock(128, kernel_size=3, padding=1, shortcut=True)
        self.res_256 = ConvolutionalBlockRes(in_channels=128, out_channels=256, kernel_size=3, padding=1, shortcut=True,
                                             pool_type="k_max")
        self.id_256 = ConvolutionalIdentityBlock(256, kernel_size=3, padding=1, shortcut=True)
        self.res_512 = ConvolutionalBlockRes(in_channels=256, out_channels=512, kernel_size=3, padding=1, shortcut=True,
                                             pool_type="k_max")
        self.id_512 = ConvolutionalIdentityBlock(512, kernel_size=3, padding=1, shortcut=True)
        self.linear_1 = nn.Linear(8 * 512, 2048)
        self.batch_norm_l1 = nn.BatchNorm1d(2048)
        self.linear_2 = nn.Linear(2048, 2048)
        self.batch_norm_l2 = nn.BatchNorm1d(2048)
        self.linear_3 = nn.Linear(2048, n_classes)

    def forward(self, inp):
        # [batch_size, sent_length]
        embedded = self.embedding(inp)
        # [batch_size, sent_lenght, emb_dim]
        embedded = embedded.permute(0, 2, 1)
        # [batch_size, emb_dim, sent_length]
        out = F.relu(self.batch_norm_conv_64(self.conv_64(embedded)))
        # [batch_size, 64, sent_length]
        out = self.id_64(out)
        # [batch_size, 64, sent_length]
        out = self.res_128(out)
        # [batch_size, 128, sent_length/2]
        out = self.id_128(out)
        # [batch_size, 128, sent_length/2]
        out = self.res_256(out)
        # [batch_size, 256, sent_length/4]
        out = self.id_256(out)
        # [batch_size, 256, sent_length/4]
        out = self.res_512(out)
        # [batch_size, 512, sent_length/8]
        out = self.id_512(out)
        # [batch_size, 512, sent_length/8]
        out = downsample_k_max_pool(out, k=8, dim=2)[0]
        out = out.reshape(out.shape[0], -1)
        # [batch_size, 512, 8]
        out = F.relu(self.batch_norm_l1(self.linear_1(out)))
        # [batch_size, 4096]
        out = F.relu(self.batch_norm_l2(self.linear_2(out)))
        # [batch_size, 512, 2048
        out = self.linear_3(out)
        # [batch_size, n_class]
        return out
