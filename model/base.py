import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    #含有@abstractmethod修饰的父类不能实例化，但是继承的子类必须实现@abstractmethod装饰的方法
    @abstractmethod
    def forward(self, *inputs):
        raise NotADirectoryError

    def __str__(self):
        """
                Model prints with number of trainable parameters
                """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)