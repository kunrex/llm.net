import csv
import os, os.path

from abc import ABC
from lib2to3.pgen2.tokenize import tokenize

from src.core.tensors.tensor import Tensor
from src.core.transformer import Transformer

class EmbeddingTransformer(Transformer, ABC):
    def __init__(self, vector_in, block_count, attention_space, ml_perceptron_space):
        super().__init__(vector_in, block_count, attention_space, ml_perceptron_space)

    # the simplest tokeniser possible
    def tokenise(self, word):
        i = 0
        current = [0 for i in range(self._vector_in)]
        for c in word:
            current[ord(c)] = 1
            current[-1] = i
            i += 1

        return current

    def train(self, file_path):
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]

        for path in files:
            if path.endswith("csv"):
                file = open(os.path.join(file_path, path))
                reader = csv.reader(file)

                main_token = None
                for row in reader:
                    main_token = Tensor.from_tensor(self.tokenise(row))
                    break

                for row in reader:
                    self._train(Tensor.from_array(self.tokenise(row)), main_token)

                file.close()

    def test(self, in_value):
        return self._test(Tensor.from_array(self.tokenise(in_value)))