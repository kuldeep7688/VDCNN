import math
import torch.nn as nn
import torch.nn.functional as F


def downsample_max_pool(x, kernel_size, stride):
    pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=1)
    return pool(x)


def downsample_k_max_pool(inp, k, dim):
    return inp.topk(k, dim)


class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class ConvolutionalBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, shortcut=False, pool_type="max_pool"):
        super().__init__()
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1,
                                bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(out_channels)
        self.conv_2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1,
                                bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(out_channels)

        if shortcut is True:
            self.conv_res = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2,
                                      bias=False)
            self.batch_norm_res = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(self.batch_norm_1(out))

        out = self.conv_2(out)
        out = F.relu(self.batch_norm_2(out))

        # downsampled
        if self.pool_type == "k_max":
            k_ = math.ceil(out.shape[2] / 2.0)
            out = downsample_k_max_pool(out, k=k_, dim=2)[0]
        else:
            out = downsample_max_pool(out, 3, 2)

        if self.shortcut is True:
            residual = self.conv_res(x)
            residual = F.relu(self.batch_norm_res(residual))
            out = out + residual
        return out


class ConvolutionalIdentityBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=1, shortcut=False):
        super().__init__()

        self.shortcut = shortcut
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1,
                                bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(in_channels)
        self.conv_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1,
                                bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(self.batch_norm_1(out))

        out = self.conv_2(out)
        out = F.relu(self.batch_norm_2(out))

        if self.shortcut is True:
            out = out + x
        else:
            out = out

        return out


class KMaxPool(nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        self.k = k
        self.dim = dim

    def forward(self, x):
        return x.topk(self.k, self.dim)[0]


def get_vdcnn(depth, embedding_dim, vocab_size, n_classes, shortcut=False, pool_type="max_pool",
              final_k_max_k=8, linear_layer_size=2048, use_batch_norm_for_linear=False,
              linear_dropout=0.4):
    # preparing model according to depth
    depth_dict = {
        9: [1, 1, 1, 1],
        17: [2, 2, 2, 2],
        29: [5, 5, 2, 2],
        49: [8, 8, 5, 3]
    }
    layer_depth_list = depth_dict[depth]

    model_layer_list = []

    model_layer_list.append(nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size))
    #     model_layer_list.append(nn.BatchNorm1d(embedding_dim))
    model_layer_list.append(Permute())
    model_layer_list.append(nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1, bias=False))
    model_layer_list.append(nn.BatchNorm1d(64))

    # 64 feature conv and res blocks
    # identity_blocs
    for i in range(0, layer_depth_list[0] - 1):
        model_layer_list.append(
            ConvolutionalIdentityBlock(64, kernel_size=3, padding=1, shortcut=shortcut)
        )
    # res blocks
    model_layer_list.append(
        ConvolutionalBlockRes(in_channels=64, out_channels=128,
                              kernel_size=3, padding=1, shortcut=shortcut,
                              pool_type=pool_type)
    )

    # 128 feature conv and res blocks
    # identity_blocks
    for i in range(0, layer_depth_list[0] - 1):
        model_layer_list.append(
            ConvolutionalIdentityBlock(128, kernel_size=3, padding=1, shortcut=shortcut)
        )
    # res blocks
    model_layer_list.append(
        ConvolutionalBlockRes(in_channels=128, out_channels=256,
                              kernel_size=3, padding=1, shortcut=shortcut,
                              pool_type=pool_type)
    )

    # 256 feature conv and res blocks
    # identity_blocks
    for i in range(0, layer_depth_list[0] - 1):
        model_layer_list.append(
            ConvolutionalIdentityBlock(256, kernel_size=3, padding=1, shortcut=shortcut)
        )
    # res blocks
    model_layer_list.append(
        ConvolutionalBlockRes(in_channels=256, out_channels=512,
                              kernel_size=3, padding=1, shortcut=shortcut,
                              pool_type=pool_type)
    )

    # 512 feature conv and res blocks
    # identity_blocks
    for i in range(0, layer_depth_list[0] - 1):
        model_layer_list.append(
            ConvolutionalIdentityBlock(512, kernel_size=3, padding=1, shortcut=shortcut)
        )
    model_layer_list.append(
        ConvolutionalIdentityBlock(512, kernel_size=3, padding=1, shortcut=False)
    )

    model_layer_list.append(KMaxPool(k=final_k_max_k, dim=2))
    model_layer_list.append(Reshape())

    if use_batch_norm_for_linear:
        model_layer_list.append(nn.Linear(final_k_max_k * 512, linear_layer_size, bias=False))
        model_layer_list.append(nn.BatchNorm1d(linear_layer_size))
        model_layer_list.append(nn.ReLU())
        model_layer_list.append(nn.Linear(linear_layer_size, linear_layer_size, bias=False))
        model_layer_list.append(nn.BatchNorm1d(linear_layer_size))
        model_layer_list.append(nn.ReLU())
        model_layer_list.append(nn.Linear(linear_layer_size, n_classes))
    else:
        model_layer_list.append(nn.Linear(final_k_max_k * 512, linear_layer_size))
        model_layer_list.append(nn.ReLU())
        model_layer_list.append(nn.Dropout(linear_dropout))
        model_layer_list.append(nn.Linear(linear_layer_size, linear_layer_size))
        model_layer_list.append(nn.ReLU())
        model_layer_list.append(nn.Dropout(linear_dropout))
        model_layer_list.append(nn.Linear(linear_layer_size, n_classes))

    model = nn.Sequential(
        *model_layer_list
    )
    return model
