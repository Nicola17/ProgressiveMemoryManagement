cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t

#####################################
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool #not sure about the performance

#####################################
cdef extern from "pmm/view.h" namespace "pmm":
    cdef cppclass View[T]:
        View()
        T& operator[](vector[uint64_t]&)
        T& dataAtSubSpace(vector[uint64_t]&)
        void setDataAtSubSpace(vector[uint64_t]&, T y)
        const vector[vector[uint64_t]]& idx_ind()
        shared_ptr[View[T]] getView(const vector[vector[uint64_t]]&)
        shared_ptr[View[T]] getViewSubSpace(const vector[vector[uint64_t]]&)
        uint64_t size(uint64_t)
