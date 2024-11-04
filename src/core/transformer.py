from abc import abstractmethod

from src.core.component import Component
from src.core.tensors.tensor import Tensor
from src.core.components.block import Block

#CROSS ENTROPY
def cost_function(predicted, actual):
    return Tensor.negate(Tensor.transpose(Tensor.log(actual)) * predicted)

#My implementation of a general purpose transformer.

#VECTOR_IN: size of each input vector (length of the 1D vector for a token)
#BLOCK_COUNT: number of blocks (number of attention + MLP groups)
#ATTENTION_SPACE: size of the query - key space in attention blocks (length of the 1D vector for a token)
#ML_PERCEPTRON_SPACE: number of tokens in an input tensor, also equal to the number of layers of the MLP
class Transformer(Component):
    device = None
    @staticmethod
    def set_device(device):
        Transformer.device = device

    def __init__(self, vector_in, block_count, attention_space, ml_perceptron_space):
        super().__init__(vector_in)
        self.__token_count = ml_perceptron_space

        self.__block_count = block_count
        self.__blocks = [Block(vector_in, attention_space, ml_perceptron_space) for i in range(block_count)]

    @abstractmethod
    def train(self, file_path):
        return

    @abstractmethod
    def test(self, in_value):
        return

    def _train(self, tensor, actual):
        result = self.__front_propagate(tensor)

        last = Tensor.soft_max(result).last()
        cost = cost_function(last, actual)

        #backpropagate
        cost.backward()

    #takes in a pytorch tensor
    def _test(self, tensor):
        result = self.__front_propagate(tensor)

        return Tensor.soft_max(result).last()

    def __front_propagate(self, tensor_in):
        current = tensor_in

        for block in self.__blocks:
            current = block.front_propagate(current)

        return current
