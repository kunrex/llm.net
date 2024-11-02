import math

import torch

from src.core.transformer import Transformer

class Tensor:
    @staticmethod
    def transpose(a):
        return Tensor.from_tensor(torch.transpose(a.__tensor, dim0 = 0, dim1 = 1))

    @staticmethod
    def soft_max(tensor_in):
        tensor = tensor_in.__tensor

        for i in range(tensor.columns()):
            s = 0
            for j in range(tensor.rows()):
                tensor[j, i] = math.exp(tensor[j, i].item())
                s += tensor[j, i]

            for j in range(tensor.rows()):
                tensor[j, i] /= s

    @staticmethod
    def relu(tensor_in):
        tensor = tensor_in.__tensor

        for i in range(tensor.size(dim = 0)):
            for j in range(tensor.size(dim = 1)):
                if tensor[i, j].item() < 0:
                    tensor[i, j] = 0

    @staticmethod
    def log(tensor_in):
        tensor = tensor_in.__tensor

        for i in range(tensor.size(dim = 0)):
            for j in range(tensor.size(dim = 1)):
                tensor[i, j] = math.log(tensor[i, j].item())

    @classmethod
    def zeros(cls, rows, columns):
        return cls().__random_init(rows, columns)

    @classmethod
    def from_value(cls, value):
        return cls().__from_py_tensor(torch.tensor(value))

    @classmethod
    def from_random(cls, rows, columns):
        return cls().__random_init(rows, columns)

    @classmethod
    def from_tensor(cls, py_tensor):
        return cls().__from_py_tensor(py_tensor)

    def __init__(self):
        self._rows = 0
        self._columns = 0

        self.__tensor = None

    def __zeros(self, rows, columns):
        if self.__tensor is None:
            self._rows = rows
            self._columns = columns

            self.__tensor = torch.zeros(self._rows, self._columns, device = Transformer.device, requires_grad = True)

        return self

    def __random_init(self, rows, columns):
        if self.__tensor is None:
            self._rows = rows
            self._columns = columns

            torch.manual_seed(2024)
            self.__tensor = torch.rand(self._rows, self._columns, device = Transformer.device, requires_grad = True)

        return self

    def __from_py_tensor(self, tensor):
        if self.__tensor is None:
            self._rows = tensor.size(dim = 0)
            self._columns = tensor.size(dim = 1)

            self.__tensor = tensor

    def rows(self):
        return self._rows

    def columns(self):
        return self._columns

    def return_tensor(self):
        tensor = self.__tensor

        self.__tensor = None
        return self.__tensor

    def __add__(self, other):
        return Tensor.from_tensor(self.__tensor + other.__tensor)

    def __mul__(self, other):
        return Tensor.from_tensor(self.__tensor @ other.__tensor)

    def __iadd__(self, other):
        self.__tensor += other.__tensor

    def __getitem__(self, item):
        shape = len(item)
        if shape == 1:
            return Tensor.from_tensor(self.__tensor[:, item[0]])
        elif shape == 2:
            return self.__tensor[item[0]][item[1]].item()

    def __setitem__(self, item, value):
        self.__tensor[item[0], item[1]] = value

    def backward(self):
        if self._rows == 1 and self._columns == 1:
            self.__tensor.backward()
