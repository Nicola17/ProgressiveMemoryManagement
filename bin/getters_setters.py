import pptable
import numpy as np

shape = np.array([30,30,30],dtype=np.uint)
chunk_shape = np.array([10,10,10],dtype=np.uint)

memory = pptable.ChunkedMemory()
memory.initialize(shape,chunk_shape, np.int32)


idx = np.array([0,0,0],dtype=np.uint)
memory.getDataAt(idx)
type(memory.getDataAt(idx))


memory.setDataAt(idx,42)

print(memory[0])
print(memory[[0,0,0]])
print(memory[np.array([0,0,0],dtype=np.uint)])


# memory[[0,0,0]] = 22
print(memory[[0,0,0]])
memory[np.array([0,0,0],dtype=np.uint)] = 33
print(memory[np.array([0,0,0],dtype=np.uint)])

print(len(memory))
print(memory[0:10:2].shape)
print(memory[0,0,0])

print(memory[:,:,0][0,0]) #TODO
print(memory[:,:,1][0,0])
# print(memory[1,:,0][0])

print(memory.shape)
print(memory[:,:,0].shape)
print(memory[1,:,0].shape)
print(memory[1,0:10:2,0].shape)

print(memory[0].shape)
print(memory[0].shape)

print("A")
print(memory[0][0].shape)
print("B")
print(memory[0][0][0])
print("C")
print(memory[0][10:].shape)
print("D")
print(memory[0][:][0].shape)


l = [0,0,0]
print(memory[l])
l = [0,0]
print(memory[l].shape)
