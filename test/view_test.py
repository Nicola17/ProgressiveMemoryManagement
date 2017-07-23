
from unittest import TestCase
from nose.tools import assert_equal
from nose.tools import assert_true
import array

import pptable
import numpy as np

class ViewTest(TestCase):
    def setUp(self):
        self.memory = pptable.ChunkedMemory()
        self.shape = np.array([30,30,30],dtype=np.uint)
        self.chunk_shape = np.array([10,10,10],dtype=np.uint)
        self.memory.initialize(self.shape,self.chunk_shape,np.float)
        self.nparray = np.zeros(self.shape)

        for z in range(self.shape[0]):
            for y in range(self.shape[1]):
                for x in range(self.shape[2]):
                    self.memory[z,y,x] = z + x + y
                    self.nparray[z,y,x] = z + x + y

    def test_content(self):
        for z in range(self.shape[0]):
            for y in range(self.shape[1]):
                for x in range(self.shape[2]):
                    assert_equal(self.memory[z,y,x], self.nparray[z,y,x])

        assert_true(np.array_equal(self.nparray, self.memory.toNpArray()))

    def test_slicing_shape(self):
        assert_true(isinstance(self.memory[:,:,0],pptable.View))
        assert_equal(self.memory[:,:,0].shape, (30,30))
        assert_equal(self.memory[1,:,0].shape, (30,))
        assert_equal(self.memory[1,0:10:2,:].shape, (5,30))
        assert_equal(self.memory[0].shape, (30,30))

    def test_np_shape_comparison(self):
        assert_equal(self.memory.shape, self.nparray.shape)
        assert_equal(self.memory.ndim, self.nparray.ndim)
        assert_equal(self.memory[:,:,0].shape, self.nparray[:,:,0].shape)
        assert_equal(self.memory[1,:,0].shape, self.nparray[1,:,0].shape)
        assert_equal(self.memory[1,0:10:2,:].shape, self.nparray[1,0:10:2,:].shape)
        assert_equal(self.memory[0].shape, self.nparray[0].shape)
        assert_equal(len(self.memory), len(self.nparray))
        assert_equal(len(self.memory[0]), len(self.nparray[0]))
        assert_equal(self.memory[0][0][0], self.nparray[0][0][0])
        assert_equal(self.memory[1][:][0].shape, self.nparray[1][:][0].shape)

    def test_np_comparison(self):
        assert_true(np.array_equal(self.memory[:,:,0].toNpArray(), self.nparray[:,:,0]))
        assert_true(np.array_equal(self.memory[1,:,0].toNpArray(), self.nparray[1,:,0]))
        assert_true(np.array_equal(self.memory[1,0:10:2,:].toNpArray(), self.nparray[1,0:10:2,:]))
        assert_true(np.array_equal(self.memory[0].toNpArray(), self.nparray[0]))
        assert_true(np.array_equal(self.memory[0][0].toNpArray(), self.nparray[0][0]))
        assert_true(np.array_equal(self.memory[1][:].toNpArray(), self.nparray[1][:]))

    def test_setters_on_memory(self):
        self.nparray[0,0,0] = 42
        self.memory[0,0,0] = 42
        assert_true(np.array_equal(self.nparray, self.memory.toNpArray()))
        self.nparray[1] = 33
        self.memory[1] = 33
        assert_true(np.array_equal(self.nparray, self.memory.toNpArray()))
        self.nparray[:,0,0] = 55
        self.memory[:,0,0] = 55
        assert_true(np.array_equal(self.nparray, self.memory.toNpArray()))
        self.nparray[0:10:2] = 22
        self.memory[0:10:2] = 22
        assert_true(np.array_equal(self.nparray, self.memory.toNpArray()))
        self.nparray[:,1] = 11
        self.memory[:,1] = 11
        assert_true(np.array_equal(self.nparray, self.memory.toNpArray()))
        self.nparray[:,1,0] = 44
        self.memory[:,1,0] = 44
        assert_true(np.array_equal(self.nparray, self.memory.toNpArray()))

    def test_setters_on_view(self):
        np_view = self.nparray[0]
        pt_view = self.memory[0]
        np_view[0,0] = 11
        pt_view[0,0] = 11
        assert_true(np.array_equal(np_view, pt_view.toNpArray()))
        np_view[:,0] = 22
        pt_view[:,0] = 22
        assert_true(np.array_equal(np_view, pt_view.toNpArray()))
        pt_view[0:10:2,0] = 33
        np_view[0:10:2,0] = 33
        assert_true(np.array_equal(np_view, pt_view.toNpArray()))
        pt_view[0] = 44
        np_view[0] = 44
        assert_true(np.array_equal(np_view, pt_view.toNpArray()))
        pt_view[5] = 55
        np_view[5] = 55
        assert_true(np.array_equal(np_view, pt_view.toNpArray()))

    def test_view_slice_setter_0(self):
        self.memory[0,0][:] = 0
        for ii in range(self.shape[2]):
            assert_equal(self.memory[0,0][ii],0)

    def test_view_slice_setter_1(self):
        self.memory[0,0][:] = 0
        self.memory[0,0][0:self.shape[2]:5] = 1
        for ii in range(self.shape[2]):
            if(ii%5) is 0:
                assert_equal(self.memory[0,0][ii],1)
            else:
                assert_equal(self.memory[0,0][ii],0)

    def test_view_slice_setter_2(self):
        self.memory[0,0][:] = 0
        self.memory[0,0][0:self.shape[2]:5] = np.arange(self.shape[2]/5)
        for ii in range(self.shape[2]):
            if(ii%5) is 0:
                assert_equal(self.memory[0,0][ii],ii/5)
            else:
                assert_equal(self.memory[0,0][ii],0)

    def test_array(self):
        m = pptable.ChunkedMemory()
        m.initialize((100,),(400,),np.float)
        a = array.array('I',)
        pass
