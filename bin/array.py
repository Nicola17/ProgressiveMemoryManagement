import pptable
import numpy as np
import array

shape = np.array([100],dtype=np.uint)
chunk_shape = np.array([4000],dtype=np.uint)

memory = pptable.ChunkedMemory()
memory.initialize(shape,chunk_shape, np.int32)

for p in memory[:]:
    print(p)
a = array.array('I',memory[:])

for p in memory:
    print(p)
a = array.array('I',memory)
