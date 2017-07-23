
#####################################
from c_chunkedmemory cimport ChunkedMemory as CChunkedMemory
from libcpp.memory cimport shared_ptr
from c_view cimport View as CView
from libcpp.vector cimport vector

cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
    ctypedef unsigned long uint32_t
    ctypedef long long int64_t
    ctypedef long int32_t

cdef enum TypeId:
    FLOAT = 1
    DOUBLE = 2
    UINT32 = 3
    UINT64 = 4
    INT32 = 5
    INT64 = 6

#####################################
# MUST be pointers otherwise std::unordered_map will throw a floating point exception
# http://stackoverflow.com/questions/19556554/floating-point-exception-when-storing-something-into-unordered-map
# awwwwww
cdef struct CCMUnion:
    CChunkedMemory[float]*      __ccm_float
    CChunkedMemory[double]*     __ccm_double
    CChunkedMemory[uint32_t]*   __ccm_uint32
    CChunkedMemory[uint64_t]*   __ccm_uint64
    CChunkedMemory[int32_t]*    __ccm_int32
    CChunkedMemory[int64_t]*    __ccm_int64

cdef class ChunkedMemory:

    #initializes the memory
    cpdef initializeNDArray(self, unsigned long[:] shape, unsigned long[:] chunk_shape) except +
    cpdef initializeTuple(self, tuple shape, tuple chunk_shape) except +
    cpdef resizeNDArray(self, unsigned long[:] shape) except +
    cpdef resizeTuple(self, tuple shape) except +

    #CRITICAL
    cpdef setDataAt(self, unsigned long[:] idx, v)
    cpdef getDataAt(self, unsigned long[:] idx)
    cdef __setDataAt1D(self, unsigned long long idx, v)
    cpdef toNpArray(self)

    #NOT critical
    cpdef getViewFromTuple(self, x)
    cpdef getViewFromInteger(self, x)
    cpdef getViewFromSlice(self, x)

    ##### Data
    # cdef CChunkedMemory[float] *thisptr  # hold a C++ instance which we're wrapping
    cdef CCMUnion __data_union
    cdef int __dtype
    cdef vector[uint64_t] __shape_vec
    cdef vector[uint64_t] __chunk_shape_vec

    ##### Cache
    cdef vector[uint64_t] __idx_vec
    cdef vector[vector[uint64_t]] __ind_idx

    ##### Dynamic typing
    cdef __setType(self, dtype)
    cdef __setShape(self)
    cdef __reshape(self)
    cdef __size(self, uint64_t x)
    cdef __getView(self)
    cdef __getData(self)
    cdef __setData(self, y)
    cdef __length(self)


#####################################
cdef struct CTypedViews:
    shared_ptr[CView[float]]      __cv_float
    shared_ptr[CView[double]]     __cv_double
    shared_ptr[CView[uint32_t]]   __cv_uint32
    shared_ptr[CView[uint64_t]]   __cv_uint64
    shared_ptr[CView[int32_t]]    __cv_int32
    shared_ptr[CView[int64_t]]    __cv_int64

cdef class View:
    cpdef getDataAt(self, unsigned long[:] idx)
    cpdef setDataAt(self, unsigned long[:] idx, float y)

    #NOT critical
    cpdef getViewFromTuple(self, x)
    cpdef getViewFromInteger(self, x)
    cpdef getViewFromSlice(self, x)

    cpdef toNpArray(self)

    ##### Data
    cdef CTypedViews __typed_views
    cdef int __dtype
    cdef vector[uint64_t] __shape_vec

    ##### Cache
    cdef vector[uint64_t] __idx_vec
    cdef vector[vector[uint64_t]] __ind_idx

    ##### Dynamic typing
    cdef __getData(self)
    cdef __setData(self, y)
    cdef __getView(self)
