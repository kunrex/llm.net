import torch, csv

from src.core.transformer import Transformer

from src.core.implementations.gpt import GPT
from src.core.implementations.embedding import EmbeddingTransformer

if torch.backends.mps.is_available():
    Transformer.set_device("mps")

dimensions = 2001
maximum_token = 200

embedding = EmbeddingTransformer(dimensions, 1, 128, maximum_token)
gpt = GPT(dimensions, 1, 128, maximum_token, embedding)

embedding.train("your/csv/directory")
gpt.train("your/csv/file.path")

# do cool things :D