import torch, csv

from src.core.transformer import Transformer

if torch.backends.mps.is_available():
    Transformer.set_device("mps")

embedding_vector_in = 2000

#the simplest tokeniser possible
def tokenise(word):
    i = 0
    current = [0 for i in range(embedding_vector_in)]
    for c in word:
        current[ord(c)] = 1
        current[-1] = i
        i += 1

    return current

#using the transformer to create an embedding transformer that works on UTF8 characters where each word has a maximum length of 128 characters
def embedding_transformer():
    file = open("/your/csv/file/path")
    reader = csv.reader(file)

    maximum_tokens = 128

    #map characters out to a 128 + 1 length array (1 value for the index of the character)
    transformer = Transformer(embedding_vector_in, 1, 128, maximum_tokens)
    for line in reader:
        for word in line:
            if len(word) > maximum_tokens:
                continue

            transformer.train(torch.tensor(tokenise(word)))

    file.close()
    return transformer

#using the transformer to create a gpt that works on lines of maximum length
def gpt():
    embedder = embedding_transformer()

    file = open("/your/csv/file/path")
    reader = csv.reader(file)

    maximum_words = 200

    transformer = Transformer(embedding_vector_in, 1, 128, maximum_words)

    for line in reader:
        current = []

        if len(line) > maximum_words:
            continue

        for word in line:
            result = embedder.test(torch.tensor(tokenise(word)))
            current.append(result.return_tensor().item())

        transformer.train(torch.tensor(current))

    file.close()
    return transformer