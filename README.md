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
transformer.train(input_tensor, expected)
```
1. `input_tensor`: A tensor representing the vector inputs to the transformer
2.  `expected`: A tensor representing the expected output in a normalized row matrix.

> Testing
```py
result = transformer.test(input_tensor)
```
1. `input_tensor`: A pytorch tensor represing the vector inputs to the transformer

## So whats happening?
Now heres the thing, I dont have the resources to train a model that produces anything especially considering my track record of starting projects a day before the deadline.
Regardless I'm gonna try to explain what goes on under the hood of a transformer so it makes sense to someone using this/going through my code.

### Embeddings
Embeddings refer to vectors (usually of a higher dimension) that are fed into a transformer. Essentially we convert human readable input into a bunch of numbers
```py
'hello` -> [104, 101, 108, 108, 111]
```
of course in the above example I've simply used the ASCII values, but that gets the point across. Essentially we take input and tokenise it (the proces of splitting into smaller sub parts or `tokens`) and embed those tokens as vectors of higher dimensions. This is done through a transformer or a nerual network. Usually though, this requires some input from a programmer (well to gather data)

### Attention Block
The attention block of a transformer is what handles context. think of it in terms of 3 matrices
1. The Query Matrix
2. The Key Matrix
3. The Value matrix
   

#### -> Query Matrix
This is a matrix that encodes all sorts of relevant queries as, you guessed it, vectors! So consider the word `dog`. We can describe this using a bunch of words: big, small, cute, adorable etc etc. A query matrix stores this query as a matrix: essentially we multiply an embedding with the matrix and obtain the result of the query, we do this for all the embeds to see how all the embeds are affected by the queries.

In practice a query matrix "stores" mutiples questions within it. These can include "am I golden" or "do I have fur"? 

#### -> Key Matrix
The matrix product of the key matrix and an embed gives us the answer to our queries. the key matrix helps an embed answer our query.

```py
queries = query_matrix * tensor_in
keys  = key_matrix * tensor_in
``` 

now we have queries and answers but we need to map out which queries correspond to which answers. This is done by taking the dot product between `queries` and `keys`. The vector dot product tells us how much of one vector does another vector contain.

Consider the query `Am I golden`. The key for embed `retriever` should have a higher correspondance to this query compared to the key for embed `pug`. In essence our dot product would be higher for the key of the embed `golden` than it would be for `pug`. 

This is what transformers define as context, or basically this is how transformers obtain context.

As a good measure we take softmax of the dot product to ensure that these values are normalized. 

#### -> Value Matrix
The value matrix is a matrix is used to amplify the effect of context. Since the any dot product after soft max can have a maximum value of 1, we use the value matrix to make context "more intense".

Now my implementation is slighly different. I dont allow embeds to gain context from embeds that follow it (just a procedure that lets me reuse training examples). So I set `dot` to an upper triangular matrix. I also split the value matrix into two matrices: `value_up` and `value_down`. 
```py
queries = query_matrix * tensor_in
keys  = key_matrix * tensor_in

dot = Tensor.softmax(Tensor.upper(Tensor.dot(keys, queries)))
result = input + value_matrix * input * dot
return result
```

### Mutlilayer Perceptron
The multilayer perceptron follows the attention layer, and here embeds do not communicate with each other. So they obtain no context from each other.

```py
result = Tensor.relu(up_project * input) * down_projection
```

This is the code implementation of an MLP. Its quite straight forward even though theres no way of explaining what it does exactly. Essentially this layer helps embeds stand on their own. Suppose after contexualising, the embed `dog` maps out to `large dog of golden colour` as a result of the context. We would want this to map to `is known as a golden retriever`. Thats where an MLP comes in. It encodes context that is based on certain facts.


And well thats it. generally we have multiple layers of attention and MLP but each of them provide small bits of context until we obtain an output vector that hopefully encodes the most appropriate response to our input. We then de-embedify this vector and present it to the user.

I hope that made sense! also thanks for checking out my project :D

Have a good day!
