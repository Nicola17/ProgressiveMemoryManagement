import pptable
import numpy as np
import math
import random
import time
import pstats, cProfile


def test_numpy(shape,n):
    print("Testing numpy...")
    start = time.time()
    memory = np.zeros(shape,dtype=np.float)
    #DO TEST
    for ii in range(n):
        memory[:] = 42
    #DONE
    end = time.time()
    print (memory)
    print("...done in " + str(end-start) + "!")
    return end-start

def test_pptable(shape,n):
    print("Testing pptable...")
    start = time.time()
    chunk_shape = np.array([1000],dtype=np.uint)
    memory = pptable.ChunkedMemory()
    memory.initialize(shape,chunk_shape,np.float)
    #DO TEST
    for ii in range(n):
        memory[:] = 42
    #DONE
    end = time.time()

    print (memory.toNpArray())
    print("...done in " + str(end-start) + "!")
    return end-start

if __name__ == "__main__":
    n = 500
    shape = np.array([100000],dtype=np.uint)

    a = test_numpy(shape,n)
    b = test_pptable(shape,n)
    print(str(b/a)+"x")

    
    cProfile.runctx("test_numpy(shape,n)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    cProfile.runctx("test_pptable(shape,n)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
