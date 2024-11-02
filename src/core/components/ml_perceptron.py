from src.core.component import Component
from src.core.tensors.tensor import Tensor

class MultilayerPerceptron(Component):
    def __init__(self, vector_in, ml_perceptron_space):
        super().__init__(vector_in)
        self.__vector_space = ml_perceptron_space

        self.__up_projection = Tensor.from_random(self._vector_in, self._vector_in)
        self.__bias_up = Tensor.from_random(self._vector_in, self.__vector_space)

        self.__down_projection = Tensor.from_random(self._vector_in, self._vector_in)
        self.bias_down = Tensor.from_random(self._vector_in, self.__vector_space)

    def front_propagate(self, tensor_in):
        projection_up = self.__up_projection * tensor_in + self.__bias_up
        Tensor.relu(projection_up)

        tensor_in += self.__down_projection * projection_up + self.bias_down