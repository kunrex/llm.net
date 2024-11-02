# llm.net
A while back I made a <a href="https://github.com/kunrex/neural.net">nerual network</a> from scratch because I was curious on the kind of black magic they are. Now that got me started on the track of recreating some common machine learning structures and so here we have my implementation of a Transformer.

Its worth mentioning, I've leveraged <a href="https://pytorch.org/">pytorch</a> here to handle most of matrix operations for me which I avoided in my neural network implementation. 

Before one comes to the conclusion, yes the project should be named transformer.net but llm.net sounds cooler so I'm keeping it.

To keep true to the name however, the `main.py` file implements the transformer to create a `word embedding generator` and, consequently, a `gpt`. Realistically it could be used to create any sort of machine learning utility based of a transformer.


## Technicalities

> Creating a Transformer
```py
#import statement
from src.core.transformer import Transformer

#create a transformer
transformer = Transformer(vector_in, block_count, attention_space, ml_perceptron_space)
```

1. `vector_in`: The length of a 1D token vector after embedding, a tensor input is a sequence of such 1D embeddings
2. `block_count`: A block is defined as an attention block + multilayer perceptron block. this parameter controls the number of blocks in the transformer
3. `attention_space`: The dimensions of the attention vector space created in attention blocks of the transformer
4. `ml_perceptron_space`: Refers to the number of tokens in the input tensor (or the number of 1D embeddings in a single input tensor)

> Training
```py
transformer.train(input_tensor)
```
1. `input_tensor`: A pytorch tensor represing the vector inputs to the transformer

> Testing
```py
result = transformer.test(input_tensor)
```
1. `input_tensor`: A pytorch tensor represing the vector inputs to the transformer
2. `result`: A 1D vector representing the output vector of the transformer
