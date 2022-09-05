import torch.nn as nn


class ShapeCheck(nn.Module):

    def __init__(self, id):
        super(ShapeCheck, self).__init__()
        self.id = id

    def forward(self, x):
        print(f'[INFO] Tensor shape at checkpoint <{self.id}>:', x.shape)
        return x
