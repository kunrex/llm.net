# llm.net
A while back I made a <a href="https://github.com/kunrex/neural.net">nerual network</a> from scratch because I was curious on the kind of black magic they are. Now that got me started on the track of recreating some common machine learning structures and so here we have my implementation of a Transformer.

Its worth mentioning, I've leveraged <a href="https://pytorch.org/">pytorch</a> here to handle most of matrix operations for me which I avoided in my neural network implementation. 

Before one comes to the conclusion, yes the project should be named transformer.net but llm.net sounds cooler so I'm keeping it.

To keep true to the name however, the `main.py` file implements the transformer to create a `word embedding generator` and, consequently, a `gpt`. Realistically it could be used to create any sort of machine learning utility based of a transformer.



