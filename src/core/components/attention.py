import math

from src.core.component import Component
from src.core.tensors.tensor import Tensor

class Attention(Component):
    def __init__(self, vector_in, attention_space):
        super().__init__(vector_in)
        self.__vector_space = attention_space

        self.__query_matrix = Tensor.from_random(self.__vector_space, self._vector_in)
        self.__key_matrix = Tensor.from_random(self.__vector_space, self._vector_in)

        self.__value_up = Tensor.from_random(self._vector_in, self.__vector_space)
        self.__value_down = Tensor.from_random(self.__vector_space, self._vector_in)
        return

    def front_propagate(self, tensor_in):
        return tensor_in + self.__value_up * self.__value_down * tensor_in * Tensor.soft_max(Tensor.upper(Tensor.transpose(self.__query_matrix * tensor_in) * self.__key_matrix))