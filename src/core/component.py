from abc import ABC, abstractmethod

class Component(ABC):
    def __init__(self, vector_in):
        self._vector_in = vector_in
        return

    @abstractmethod
    def front_propagate(self, tensor_in):
        return