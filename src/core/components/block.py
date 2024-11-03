from src.core.component import Component
from src.core.components.attention import Attention
from src.core.components.ml_perceptron import MultilayerPerceptron

class Block(Component):
    def __init__(self, vector_in, attention_space, ml_perceptron_space):
        super().__init__(vector_in)
        self.__attention = Attention(vector_in, attention_space)
        self.__ml_perceptron = MultilayerPerceptron(vector_in, ml_perceptron_space)

    def front_propagate(self, tensor_in):
        return self.__ml_perceptron.front_propagate(self.__attention.front_propagate(tensor_in))