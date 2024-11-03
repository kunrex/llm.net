import torch
import torch.nn.functional as function

from src.core.transformer import Transformer

class Tensor:
    @staticmethod
    def transpose(a):
        return Tensor.from_tensor(torch.transpose(a.__tensor, dim0 = 0, dim1 = 1))

    @staticmethod
    def upper(tensor_in):
        return Tensor.from_tensor(torch.triu(tensor_in.__tensor))

    @staticmethod
    def soft_max(tensor_in):
        return Tensor.from_tensor(function.softmax(torch.triu(tensor_in.__tensor), dim = 0))

    @staticmethod
    def relu(tensor_in):
        return Tensor.from_tensor(function.relu(tensor_in.__tensor))

    @staticmethod
    def log(tensor_in):
        return Tensor.from_tensor(torch.log(tensor_in.__tensor))

    @classmethod
    def zeros(cls, rows, columns):
        return cls().__zeros(rows, columns)

    @classmethod
    def from_value(cls, value):
        return cls().__from_py_tensor(torch.tensor(value))

    @classmethod
    def from_random(cls, rows, columns):
        return cls().__random_init(rows, columns)

    @classmethod
    def from_tensor(cls, py_tensor):
        return cls().__from_py_tensor(py_tensor)

    @classmethod
    def from_array(cls, array):
        return cls().__from_py_tensor(torch.tensor(array))

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

    def __add__(self, other):
        return Tensor.from_tensor(self.__tensor + other.__tensor)

    def __mul__(self, other):
        return Tensor.from_tensor(self.__tensor @ other.__tensor)

    def get_column(self, index):
        return self.__tensor[:, index]

    def backward(self):
        if self._rows == 1 and self._columns == 1:
            self.__tensor.backward()
