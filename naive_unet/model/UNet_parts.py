import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convolution(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.convolution = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        go_down = self.convolution(x)
        pool = self.max_pool(go_down)
        return go_down, pool


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.convolution_inverse = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size,
                                                      stride=stride)
        self.convolution = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # Two input variables due to skip connections in architecture
        go_up = self.convolution_inverse(x1)
        x = torch.cat([go_up, x2], 1)
        return self.convolution(x)