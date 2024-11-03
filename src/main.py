import torch, csv

from src.core.tensors.tensor import Tensor
from src.core.transformer import Transformer

if torch.backends.mps.is_available():
    Transformer.set_device("mps")

#map words to a vector space of 2000 dimensions + 1 index to store the relative position of the character
embedding_vector_in = 2001

#the simplest tokeniser possible
def tokenise(word):
    i = 0
    current = [0 for i in range(embedding_vector_in)]
    for c in word:
        current[ord(c)] = 1
        current[-1] = i
        i += 1

    return current

#using the transformer to create a gpt that works on lines of maximum length
def gpt():
    embedder = #Use an embedding system, I couldn't implement my own :(

    file = open("/your/csv/file/path")
    reader = csv.reader(file)

    maximum_words = 200

    #one extra index to encode the relative position of the index
    transformer = Transformer(embedding_vector_in + 1, 1, 128, maximum_words)

    for line in reader:
        i = 0
        current = []
        for word in line:
            if i > maximum_words:
                i = 0
                current.clear()
                transformer.train(torch.tensor(current), Tensor.from_array(current))
                continue

            current.append(embedder.test(word))
            i += 1

        if len(current) > 0
            transformer.train(torch.tensor(current), Tensor.from_array(current))

    file.close()
    return transformer