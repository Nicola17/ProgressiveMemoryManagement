import pptable
import numpy as np
import math
import random
import time

def test_numpy(shape):
    print("Testing numpy...")
    start = time.time()
    memory = np.zeros(shape,dtype=np.float)
    #DO TEST
    test_func(memory)
    #DONE
    end = time.time()
    print("...done in " + str(end-start) + "!")

def test_pptable(shape):
    print("Testing pptable...")
    start = time.time()
    chunk_shape = np.array([1000],dtype=np.uint)
    memory = pptable.ChunkedMemory()
    memory.initialize(shape,chunk_shape,np.float)
    #DO TEST
    test_func(memory)
    #DONE
    end = time.time()
    print("...done in " + str(end-start) + "!")

def test_func(ndarray):
    idx = np.zeros(len(ndarray.shape),dtype=np.uint)
    for ii in range(ndarray.shape[0]):
        idx[0] = ii
        ndarray[idx] = math.sqrt(random.random())

if __name__ == "__main__":
    shape = np.array([1000000],dtype=np.uint)
    test_numpy(shape)
    test_pptable(shape)
