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
        queries = self.__query_matrix * tensor_in
        keys = self.__key_matrix * tensor_in

        dot = Tensor.transpose(queries) * keys
        for i in range(0, self._vector_in):
            for j in range(i + 1, self._vector_in):
                dot[(i, j)] = -math.inf

        Tensor.soft_max(dot)
        tensor_in += self.__value_up * self.__value_down * tensor_in * dot