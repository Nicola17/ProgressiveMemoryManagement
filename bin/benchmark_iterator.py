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
    batch = np.arange(shape[0])
    for ii in range(n):
        memory[:] = batch
    #DONE
    end = time.time()
    print (memory)
    print("...done in " + str(end-start) + "!")

def test_pptable(shape,n):
    print("Testing pptable...")
    start = time.time()
    chunk_shape = np.array([4000],dtype=np.uint)
    memory = pptable.ChunkedMemory()
    memory.initialize(shape,chunk_shape,np.float)
    #DO TEST
    batch = np.arange(shape[0])
    for ii in range(n):
        memory[:] = batch
    #DONE
    end = time.time()

    print (memory.toNpArray())
    print("...done in " + str(end-start) + "!")

if __name__ == "__main__":
    n = 500
    shape = np.array([100000],dtype=np.uint)

    # test_numpy(shape,n)
    # test_pptable(shape,n)

    cProfile.runctx("test_numpy(shape,n)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    cProfile.runctx("test_pptable(shape,n)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
