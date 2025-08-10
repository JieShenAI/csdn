import sys
for p in sys.path:
    print(p)

from vector_pkag.vector_base import Vector

v = Vector(1, 2)

print(v.sum_vector())

print(v.init_vector())

print(v.sum_vector())
