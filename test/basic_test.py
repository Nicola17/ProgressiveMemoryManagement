
from unittest import TestCase
from nose.tools import assert_equal

import pptable
import numpy as np

class ChunkedMemoryTest(TestCase):
    def setUp(self):
        self.memory = pptable.ChunkedMemory()
        self.shape = np.array([30,30,30],dtype=np.uint)
        self.chunk_shape = np.array([10,10,10],dtype=np.uint)

        self.memory.initialize(self.shape,self.chunk_shape,np.float)


    #tests a sigle assigment
    def test_set_data(self):
        idx = np.array([0,0,0],dtype=np.uint)
        self.memory.setDataAt(idx,42)
        assert_equal(self.memory.getDataAt(idx), 42)

    #test the assigment of all elemnts in the table
    def test_iota(self):
        idx = np.array([0,0,0],dtype=np.uint)
        iota = 0
        for z in range(self.shape[0]):
            idx[0] = z
            for y in range(self.shape[1]):
                idx[1] = y
                for x in range(self.shape[0]):
                    idx[2] = x
                    self.memory.setDataAt(idx,iota)
                    iota += 1

        iota = 0
        for z in range(self.shape[0]):
            idx[0] = z
            for y in range(self.shape[1]):
                idx[1] = y
                for x in range(self.shape[0]):
                    idx[2] = x
                    assert_equal(self.memory.getDataAt(idx), iota)
                    iota += 1

    def test_getitem(self):
        idx = np.array([0,0,0],dtype=np.uint)
        self.memory[idx]

    def test_resize(self):
        idx = np.array([0,0,0],dtype=np.uint)
        iota = 0
        for z in range(self.shape[0]):
            idx[0] = z
            for y in range(self.shape[1]):
                idx[1] = y
                for x in range(self.shape[0]):
                    idx[2] = x
                    self.memory.setDataAt(idx,iota)
                    iota += 1

        new_shape = np.array([60,60,60],dtype=np.uint)
        self.memory.resize(new_shape)

        iota = 0
        for z in range(self.shape[0]):
            idx[0] = z
            for y in range(self.shape[1]):
                idx[1] = y
                for x in range(self.shape[0]):
                    idx[2] = x
                    assert_equal(self.memory.getDataAt(idx), iota)
                    iota += 1

        for z in range(self.shape[0],new_shape[0]):
            idx[0] = z
            for y in range(self.shape[1],new_shape[1]):
                idx[1] = y
                for x in range(self.shape[0],new_shape[2]):
                    idx[2] = x
                    assert_equal(self.memory.getDataAt(idx), 0)

    def test_resize_tuple(self):
        idx = np.array([0,0,0],dtype=np.uint)
        iota = 0
        for z in range(self.shape[0]):
            idx[0] = z
            for y in range(self.shape[1]):
                idx[1] = y
                for x in range(self.shape[0]):
                    idx[2] = x
                    self.memory.setDataAt(idx,iota)
                    iota += 1

        new_shape = (60,60,60)
        self.memory.resize(new_shape)

        iota = 0
        for z in range(self.shape[0]):
            idx[0] = z
            for y in range(self.shape[1]):
                idx[1] = y
                for x in range(self.shape[0]):
                    idx[2] = x
                    assert_equal(self.memory.getDataAt(idx), iota)
                    iota += 1

        for z in range(self.shape[0],new_shape[0]):
            idx[0] = z
            for y in range(self.shape[1],new_shape[1]):
                idx[1] = y
                for x in range(self.shape[0],new_shape[2]):
                    idx[2] = x
                    assert_equal(self.memory.getDataAt(idx), 0)

    def test_tuple_init(self):
        tuple_memory = pptable.ChunkedMemory()
        tuple_memory.initialize((30,30,30),(10,10,10),np.float)
        assert_equal(self.memory.shape, tuple_memory.shape)

    def test_empty(self):
        empty_memory = pptable.ChunkedMemory()
        empty_memory.initialize((0,0,0),(10,10,10),np.float)
        assert_equal(empty_memory.shape, (0,0,0))
        assert_equal(empty_memory[:].shape, (0,0,0))
        new_shape = (60,60,60)
        empty_memory.resize(new_shape)
        assert_equal(empty_memory.shape, new_shape)

        empty_memory[0,0,0] = 33
        assert_equal(empty_memory[0,0,0], 33)
        assert_equal(empty_memory[0][0,0], 33)
        #assert_equal(empty_memory[0][0][0], 33)
