cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t

#####################################
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool #not sure about the performance

#####################################
from c_view cimport View

#####################################
cdef extern from "pmm/chunked_memory.h" namespace "pmm":
    cdef cppclass ChunkedMemory[T]:
        ChunkedMemory()
        void setShape(const vector[uint64_t]&, const vector[uint64_t]&)
        void reshape(const vector[uint64_t]&)

        T& operator[](vector[uint64_t]&) #const gives an error... mumble mumble

        shared_ptr[View[T]] getView(const vector[vector[uint64_t]]&)

        uint64_t length()
        uint64_t ndim()
        uint64_t size(uint64_t)
        const vector[uint64_t]& shape()

###### TODO ######
        void optimizeChunks()
        void test1(vector[uint64_t])
        void loadInMemory(vector[uint64_t])
        bool inMemory(vector[uint64_t])
