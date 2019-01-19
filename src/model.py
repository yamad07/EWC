from .layer import linear_layer
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.f1 = linear_layer(1, 16)
        self.f2 = linear_layer(16, 32)
        self.f4 = nn.Linear(800, 10)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.f1(x)
        h = self.f2(h)
        h = h.view(batch_size, -1)
        h = self.f4(h)
        return F.log_softmax(h, dim=1)
