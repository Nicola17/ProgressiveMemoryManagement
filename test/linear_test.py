
from unittest import TestCase
from nose.tools import assert_equal
from nose.tools import assert_true


import pptable
import numpy as np

class LinearTest(TestCase):
    def setUp(self):
        self.memory = pptable.ChunkedMemory()
        self.shape = np.array([1000],dtype=np.uint)
        self.chunk_shape = np.array([20],dtype=np.uint)
        self.memory.initialize(self.shape,self.chunk_shape,np.float)
        self.nparray = np.arange(1000)
        for ii in range(self.shape[0]):
            self.memory[ii] = ii


    #tests a sigle assigment
    def test_get_data_linear(self):
        for ii in range(self.shape[0]):
            assert_equal(self.memory[ii],ii)

    #tests a sigle assigment
    def test_np_comparison_linear(self):
        assert_equal(self.memory.shape,self.nparray.shape)
        assert_true(np.array_equal(self.memory.toNpArray(),self.nparray))
        assert_true(np.array_equal(self.memory[3:].toNpArray(),self.nparray[3:]))
        assert_true(np.array_equal(self.memory[:].toNpArray(),self.nparray[:]))
        assert_equal(self.memory[1],self.nparray[1])
        assert_true(np.array_equal(self.memory[0:1000:20].toNpArray(),self.nparray[0:1000:20]))

    def test_resize(self):
        self.memory.resize((5000,))
        for ii in range(0,1000):
            assert_equal(self.memory[ii],ii)
        for ii in range(1000,5000):
            assert_equal(self.memory[ii],0)

        assert_true(np.array_equal(self.memory[:1000].toNpArray(),self.nparray))
        assert_true(np.array_equal(self.memory[1000:].toNpArray(),np.zeros([4000])))

    def test_linear_slice_setter_0(self):
        self.memory[:] = 0
        for ii in range(self.shape[0]):
            assert_equal(self.memory[ii],0)

    def test_linear_slice_setter_1(self):
        self.memory[:] = 0
        self.memory[0:self.shape[0]:5] = 1
        for ii in range(self.shape[0]):
            if(ii%5) is 0:
                assert_equal(self.memory[ii],1)
            else:
                assert_equal(self.memory[ii],0)

    def test_linear_slice_setter_2(self):
        self.memory[:] = 0
        self.memory[0:self.shape[0]:5] = np.arange(self.shape[0]/5)
        for ii in range(self.shape[0]):
            if(ii%5) is 0:
                assert_equal(self.memory[ii],ii/5)
            else:
                assert_equal(self.memory[ii],0)
