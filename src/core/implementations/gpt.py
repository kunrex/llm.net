import csv, torch

from abc import ABC

from src.core.tensors.tensor import Tensor
from src.core.transformer import Transformer

class GPT(Transformer, ABC):
    def __init__(self, vector_in, block_count, attention_space, ml_perceptron_space, embedding):
        super().__init__(vector_in, block_count, attention_space, ml_perceptron_space)

        self.__embeddings = embedding
        self.__maximum_token_count = ml_perceptron_space

    def train(self, file_path):
        file = open(file_path)
        reader = csv.reader(file)

        for line in reader:
            i = 0
            current = []
            for word in line:
                if i > self.__maximum_token_count:
                    break

                embed = self.__embeddings.test(word)

                self._train(Tensor.transpose(Tensor.from_array(current)), embed)
                current.append(embed.raw())
                i += 1

        file.close()

    def test(self, in_value):
        return self._test(Tensor.transpose(Tensor.from_array([self.__embeddings(x).raw() for x in in_value.split(' ')[0 :self.__maximum_token_count]])))