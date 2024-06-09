import torch.nn as nn
import torch

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class SqueezeChannels(nn.Module):
    # Squeezes, in a three-dimensional tensor, the third dimension.
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class CausalConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, dropout, final=False):
        super(CausalConvolutionBlock, self).__init__()
        # First causal convolution
        conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = nn.LeakyReLU()
        # dropout1 = nn.Dropout(dropout)
        # Second causal convolution
        conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()
        # dropout2 = nn.Dropout(dropout)
        # Causal network
        self.causal = nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2)
        # Residual connection
        self.upordownsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        # Final activation function
        self.relu = nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)

class CausalCNN(nn.Module):
    def __init__(self, in_channels, channels, depth, out_channels, kernel_size, dropout):
        super(CausalCNN, self).__init__()
        layers = []  # List of causal convolution blocks
        for i in range(depth):
            dilation_size = 2 ** i
            in_channels_block = in_channels if i == 0 else channels
            layers += [
                CausalConvolutionBlock(in_channels_block, channels, kernel_size, (kernel_size - 1) * dilation_size,
                                       dilation_size, dropout)]
        # Last layer
        dilation_size = 2 * dilation_size
        layers += [CausalConvolutionBlock(channels, out_channels, kernel_size, (kernel_size - 1) * dilation_size,
                                          dilation_size, dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CausalCNNEncoder(nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size, dropout=0.2):
        super(CausalCNNEncoder, self).__init__()
        self.network = nn.Sequential(
            CausalCNN(in_channels, channels, depth, reduced_size, kernel_size, dropout),
            nn.AdaptiveMaxPool1d(1),
            SqueezeChannels(),  # Squeezes the third dimension (time)
            nn.Linear(reduced_size, out_channels),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)