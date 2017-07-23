import pptable
import numpy as np
import math
import random
import time

def test_numpy(n,shape):
    print("Testing numpy...")
    start = time.time()
    memory = np.zeros(shape,dtype=np.float)
    np.resize(memory,shape*2)
    #DO TEST
    test_func(n,memory[0:shape[0]:10])
    #DONE
    end = time.time()
    print("...done in " + str(end-start) + "!")

def test_pptable(n,shape):
    print("Testing pptable...")
    start = time.time()
    chunk_shape = np.array([1000],dtype=np.uint)
    memory = pptable.ChunkedMemory()
    memory.initialize(shape,chunk_shape,np.float)
    memory.resize(shape*2)
    #DO TEST
    test_func(n,memory[0:shape[0]:10])
    #DONE
    end = time.time()
    print("...done in " + str(end-start) + "!")

def test_func(n,ndarray):
    idx = np.zeros(len(ndarray.shape),dtype=np.uint)
    for ii in range(n):
        for ss in range(len(idx)):
            idx[ss] = random.randrange(ndarray.shape[ss])
        ndarray[idx] = math.sqrt(random.random())

if __name__ == "__main__":
    n = 50000
    shape = np.array([1000000/2],dtype=np.uint)
    test_numpy(n,shape)
    test_pptable(n,shape)
