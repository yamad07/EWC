import torch.nn as nn

def linear_layer(input_size, output_size):
    return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2)
            )
