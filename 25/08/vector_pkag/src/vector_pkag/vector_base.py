from dataclasses import dataclass

from .init_vector import init_vector as init_v
from vector_pkag.sum_vector import sum_vector as sum_v

@dataclass
class Vector:
    x : int
    y : int
    
    def add(self, other):
        self.x += other.x
        self.y += other.y
        
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def init_vector(self):
        return init_v(self)

    def sum_vector(self)->int:
        return sum_v(self)
        