import pptable
import numpy as np
import math
import random
import time
import pstats, cProfile


def test_numpy(shape,n,batch_size):
    print("Testing numpy...")
    start = time.time()
    memory = np.zeros(shape,dtype=np.float)
    #DO TEST
    batch = np.arange(batch_size)
    for ii in range(n):
        old_size = memory.shape[0]
        memory = np.resize(memory,tuple([old_size+batch_size]))
        # print (memory.shape)
        memory[old_size:] = batch
    print (memory)
    #DONE
    end = time.time()
    print("...done in " + str(end-start) + "!")

def test_pptable(shape,n,batch_size):
    print("Testing pptable...")
    start = time.time()
    chunk_shape = np.array([4000],dtype=np.uint)
    memory = pptable.ChunkedMemory()
    memory.initialize(shape,chunk_shape,np.float)
    #DO TEST
    batch = np.arange(batch_size)
    for ii in range(n):
        old_size = memory.shape[0]
        memory.resize(tuple([old_size+batch_size]))
        memory[old_size:] = batch
    print (memory)
    #DONE
    end = time.time()
    print("...done in " + str(end-start) + "!")

if __name__ == "__main__":
    n = 1000
    batch_size = 1000
    shape = np.array([0],dtype=np.uint)

    cProfile.runctx("test_numpy(shape,n,batch_size)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

    cProfile.runctx("test_pptable(shape,n,batch_size)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
