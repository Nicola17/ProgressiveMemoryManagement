# distutils: language = c++
# cython: profile=True

cimport c_chunkedmemory
from c_view cimport View as CView

#TODO I'd like to isolate this guys a bit...
from libcpp.vector cimport vector
cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t

import numpy as np
from itertools import product

#define PIPPO

cdef class ChunkedMemory:
    def __cinit__(self):
        pass
        self.__data_union.__ccm_float   = new CChunkedMemory[float]()
        self.__data_union.__ccm_double  = new CChunkedMemory[double]()
        self.__data_union.__ccm_uint32  = new CChunkedMemory[uint32_t]()
        self.__data_union.__ccm_uint64  = new CChunkedMemory[uint64_t]()
        self.__data_union.__ccm_int32   = new CChunkedMemory[int32_t]()
        self.__data_union.__ccm_int64   = new CChunkedMemory[int64_t]()

    def __dealloc__(self):
        del self.__data_union.__ccm_float
        del self.__data_union.__ccm_double
        del self.__data_union.__ccm_uint32
        del self.__data_union.__ccm_uint64
        del self.__data_union.__ccm_int32
        del self.__data_union.__ccm_int64

    cdef __setType(self,dtype):
        #Saving __dtype
        if dtype is np.float:
            self.__dtype = TypeId.FLOAT
        elif dtype is np.double:
            self.__dtype = TypeId.DOUBLE
        elif dtype is np.uint32:
            self.__dtype = TypeId.UINT32
        elif dtype is np.uint64:
            self.__dtype = TypeId.UINT64
        elif dtype is np.int32:
            self.__dtype = TypeId.INT32
        elif dtype is np.int64 or np.dtype(np.int64) is dtype:
            self.__dtype = TypeId.INT64
        else:
            self.__dtype = TypeId.FLOAT
            # raise TypeError("type unknown")

    cdef __setShape(self):
        if self.__dtype == TypeId.FLOAT:
            self.__data_union.__ccm_float.setShape(self.__shape_vec,self.__chunk_shape_vec)
        elif self.__dtype == TypeId.DOUBLE:
            self.__data_union.__ccm_double.setShape(self.__shape_vec,self.__chunk_shape_vec)
        elif self.__dtype == TypeId.UINT32:
            self.__data_union.__ccm_uint32.setShape(self.__shape_vec,self.__chunk_shape_vec)
        elif self.__dtype == TypeId.UINT64:
            self.__data_union.__ccm_uint64.setShape(self.__shape_vec,self.__chunk_shape_vec)
        elif self.__dtype == TypeId.INT32:
            self.__data_union.__ccm_int32.setShape(self.__shape_vec,self.__chunk_shape_vec)
        elif self.__dtype == TypeId.INT64:
            self.__data_union.__ccm_int64.setShape(self.__shape_vec,self.__chunk_shape_vec)
        else:
            raise TypeError("type unknown")

    cdef __reshape(self):
        if self.__dtype == TypeId.FLOAT:
            self.__data_union.__ccm_float.reshape(self.__shape_vec)
        elif self.__dtype == TypeId.DOUBLE:
            self.__data_union.__ccm_double.reshape(self.__shape_vec)
        elif self.__dtype == TypeId.UINT32:
            self.__data_union.__ccm_uint32.reshape(self.__shape_vec)
        elif self.__dtype == TypeId.UINT64:
            self.__data_union.__ccm_uint64.reshape(self.__shape_vec)
        elif self.__dtype == TypeId.INT32:
            self.__data_union.__ccm_int32.reshape(self.__shape_vec)
        elif self.__dtype == TypeId.INT64:
            self.__data_union.__ccm_int64.reshape(self.__shape_vec)
        else:
            raise TypeError("type unknown")

    #TODO should be cached -> look at __shape_vec... do it later though
    cdef __size(self, uint64_t x):
        if self.__dtype == TypeId.FLOAT:
            return self.__data_union.__ccm_float.size(x)
        elif self.__dtype == TypeId.DOUBLE:
            return self.__data_union.__ccm_double.size(x)
        elif self.__dtype == TypeId.UINT32:
            return self.__data_union.__ccm_uint32.size(x)
        elif self.__dtype == TypeId.UINT64:
            return self.__data_union.__ccm_uint64.size(x)
        elif self.__dtype == TypeId.INT32:
            return self.__data_union.__ccm_int32.size(x)
        elif self.__dtype == TypeId.INT64:
            return self.__data_union.__ccm_int64.size(x)
        else:
            raise TypeError("type unknown")

    cdef __getView(self): #uint64_t doesn't work here
        view = View()
        view.__dtype = self.__dtype
        view.__ind_idx = self.__ind_idx
        for i in range(view.__ind_idx.size()):
            if view.__ind_idx[i].size() > 1:
                view.__shape_vec.push_back(view.__ind_idx[i].size())
        if view.__shape_vec.empty():
            view.__shape_vec.push_back(1)

        if self.__dtype == TypeId.FLOAT:
            view.__typed_views.__cv_float = self.__data_union.__ccm_float.getView(self.__ind_idx)
        elif self.__dtype == TypeId.DOUBLE:
            view.__typed_views.__cv_double = self.__data_union.__ccm_double.getView(self.__ind_idx)
        elif self.__dtype == TypeId.UINT32:
            view.__typed_views.__cv_uint32 = self.__data_union.__ccm_uint32.getView(self.__ind_idx)
        elif self.__dtype == TypeId.UINT64:
            view.__typed_views.__cv_uint64 = self.__data_union.__ccm_uint64.getView(self.__ind_idx)
        elif self.__dtype == TypeId.INT32:
            view.__typed_views.__cv_int32 = self.__data_union.__ccm_int32.getView(self.__ind_idx)
        elif self.__dtype == TypeId.INT64:
            view.__typed_views.__cv_int64 = self.__data_union.__ccm_int64.getView(self.__ind_idx)
        else:
            raise TypeError("type unknown")or("type unknown")
        return view

    cdef __getData(self): #uint64_t doesn't work here
        if self.__dtype == TypeId.FLOAT:
            return self.__data_union.__ccm_float[0][self.__idx_vec]
        elif self.__dtype == TypeId.DOUBLE:
            return self.__data_union.__ccm_double[0][self.__idx_vec]
        elif self.__dtype == TypeId.UINT32:
            return self.__data_union.__ccm_uint32[0][self.__idx_vec]
        elif self.__dtype == TypeId.UINT64:
            return self.__data_union.__ccm_uint64[0][self.__idx_vec]
        elif self.__dtype == TypeId.INT32:
            return self.__data_union.__ccm_int32[0][self.__idx_vec]
        elif self.__dtype == TypeId.INT64:
            return self.__data_union.__ccm_int64[0][self.__idx_vec]
        else:
            raise TypeError("type unknown")or("type unknown")

    # Type for y?
    cdef __setData(self, y): #uint64_t doesn't work here
        if self.__dtype == TypeId.FLOAT:
            self.__data_union.__ccm_float[0][self.__idx_vec] = y
        elif self.__dtype == TypeId.DOUBLE:
            self.__data_union.__ccm_double[0][self.__idx_vec] = y
        elif self.__dtype == TypeId.UINT32:
            self.__data_union.__ccm_uint32[0][self.__idx_vec] = y
        elif self.__dtype == TypeId.UINT64:
            self.__data_union.__ccm_uint64[0][self.__idx_vec] = y
        elif self.__dtype == TypeId.INT32:
            self.__data_union.__ccm_int32[0][self.__idx_vec] = y
        elif self.__dtype == TypeId.INT64:
            self.__data_union.__ccm_int64[0][self.__idx_vec] = y
        else:
            raise TypeError("type unknown")or("type unknown")
        pass

    cdef __length(self):
        if self.__dtype == TypeId.FLOAT:
            return self.__data_union.__ccm_float.length()
        elif self.__dtype == TypeId.DOUBLE:
            return self.__data_union.__ccm_double.length()
        elif self.__dtype == TypeId.UINT32:
            return self.__data_union.__ccm_uint32.length()
        elif self.__dtype == TypeId.UINT64:
            return self.__data_union.__ccm_uint64.length()
        elif self.__dtype == TypeId.INT32:
            return self.__data_union.__ccm_int32.length()
        elif self.__dtype == TypeId.INT64:
            return self.__data_union.__ccm_int64.length()
        else:
            raise TypeError("type unknown")

    def initialize(self, shape, chunk_shape, dtype):
        self.__setType(dtype)
        if isinstance(shape,tuple) and isinstance(chunk_shape,tuple):
            self.initializeTuple(shape,chunk_shape)
        elif isinstance(shape,np.ndarray) and isinstance(chunk_shape,np.ndarray):
            self.initializeNDArray(shape,chunk_shape)
        else:
            raise TypeError("unknown type")

    #Check if this is better!
    #https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
    cpdef initializeNDArray(self, unsigned long[:] shape, unsigned long[:] chunk_shape) except +:
        #NB: NOT critical for performance
        self.__shape_vec.clear()
        self.__chunk_shape_vec.clear()
        for i in xrange(len(shape)):
            self.__shape_vec.push_back(shape[i])
        for i in xrange(len(chunk_shape)):
            self.__chunk_shape_vec.push_back(chunk_shape[i])
        self.__setShape()


    cpdef initializeTuple(self, tuple shape, tuple chunk_shape) except +:
        #NB: NOT critical for performance
        self.__shape_vec.clear()
        self.__chunk_shape_vec.clear()
        for i in range(len(shape)):
            self.__shape_vec.push_back(shape[i])
        for i in range(len(chunk_shape)):
            self.__chunk_shape_vec.push_back(chunk_shape[i])
        self.__setShape()

    def resize(self, shape):
        if isinstance(shape,tuple):
            self.resizeTuple(shape)
        elif isinstance(shape,np.ndarray):
            self.resizeNDArray(shape)
        else:
            raise TypeError("unknown type")

    cpdef resizeNDArray(self, unsigned long[:] shape) except +:
        self.__shape_vec.clear()
        for i in xrange(len(shape)):
            self.__shape_vec.push_back(shape[i])
        self.__reshape()

    cpdef resizeTuple(self, tuple shape) except +:
        self.__shape_vec.clear()
        for i in xrange(len(shape)):
            self.__shape_vec.push_back(shape[i])
        self.__reshape()

    #########################
    cpdef getDataAt(self, unsigned long[:] idx):
        self.__idx_vec.resize(self.__shape_vec.size())
        for i in xrange(self.__shape_vec.size()):
            self.__idx_vec[i] = idx[i]
            if(self.__idx_vec[i] >= self.__shape_vec[i]):
                raise IndexError("index out of bound")
        return self.__getData()

    cpdef getViewFromTuple(self, x):
        self.__ind_idx.clear()
        self.__ind_idx.resize(self.ndim)
        for ii in range(0,len(x)):
            if(isinstance(x[ii],slice)):
                for jj in xrange(*(x[ii]).indices(self.__size(ii))):
                    self.__ind_idx[ii].push_back(jj)
            elif(isinstance(x[ii],int)):
                self.__ind_idx[ii].push_back(x[ii])
            else:
                raise TypeError("Unknown type in View creation")

        for ii in range(len(x),self.ndim):
            for jj in range(0,self.__size(ii)):
                self.__ind_idx[ii].push_back(jj) #iota

        return self.__getView()

    cpdef getViewFromInteger(self, x):
        self.__ind_idx.clear()
        self.__ind_idx.resize(self.ndim)
        self.__ind_idx[0].push_back(x)
        for ii in range(1,self.ndim):
            for jj in range(0,self.__size(ii)):
                self.__ind_idx[ii].push_back(jj) #iota

        return self.__getView()

    cpdef getViewFromSlice(self, x):
        self.__ind_idx.clear()
        self.__ind_idx.resize(self.ndim)
        for ii in xrange(*(x).indices(self.__size(0))):
            self.__ind_idx[0].push_back(ii)

        for jj in range(1,self.ndim):
            for ii in range(0,self.__size(jj)):
                self.__ind_idx[jj].push_back(ii) #iota

        return self.__getView()


    #########################
    def __getitem__(self,x):
        if self.__length() is 0:
            return self
        #Integer
        if isinstance(x, int) or isinstance(x, np.int64):
            #1D case: direct access
            if self.ndim is 1:
                a = np.asarray([x],dtype=np.uint)
                return self.getDataAt(a)
            #else I return a view
            else:
                return self.getViewFromInteger(x)

        elif isinstance(x, list):
            if len(x) > self.ndim:
                raise IndexError("too many indices for array")
            elif len(x) is self.ndim:
                #direct indexing
                np_array = np.asarray(x,dtype=np.uint)
                return self.getDataAt(np_array)
            else:
                return self.getViewFromTuple(tuple(x))

        elif isinstance(x, tuple):
            if len(x) > self.ndim:
                raise IndexError("too many indices for array")
            #if the tuple does not contain any slicer, then return a values
            #otherwise returns a view
            elif len(x) is self.ndim:
                idx = np.zeros(len(x),dtype=np.uint)
                for ii in range(0,len(x)):
                    if(isinstance(x[ii],slice)):
                        return self.getViewFromTuple(x)
                    else:
                        idx[ii] = x[ii]
                return self.getDataAt(idx)
            else:
                return self.getViewFromTuple(x)

        #TODO how efficient is this??
        elif isinstance(x, np.ndarray):
            return self.getDataAt(x)

        elif isinstance(x, slice):
            return self.getViewFromSlice(x)

        else:
            print (type(x))
            raise TypeError("type unknown")

    #########################
    cpdef setDataAt(self, unsigned long[:] idx, v):
        self.__idx_vec.resize(self.__shape_vec.size())
        for i in xrange(self.__shape_vec.size()):
            self.__idx_vec[i] = idx[i]
            if(self.__idx_vec[i] >= self.__shape_vec[i]):
                raise IndexError("index out of bound")
        self.__setData(v)

    cdef __setDataAt1D(self, unsigned long long idx, v):
        # self.__idx_vec.resize(1)
        self.__idx_vec[0] = idx
        if(self.__idx_vec[0] >= self.__shape_vec[0]):
            raise IndexError("index out of bound")
        self.__setData(v)

    def __setDataFromTuple(self,x,y):
        #NB: CRITICAL -> cache the vector
        if len(x) < self.ndim :
            self.__setHyperplaneFromTuple(self,x,y)
        else:
            a = np.zeros(self.shape,dtype=np.float)
            self.setDataAt(a,y)


    #########################
    def __setitem__(self,x,y):
        cdef uint64_t ii
        if isinstance(x, int) or isinstance(x, np.int64):
            if self.ndim is 1:
                a = np.asarray([x],dtype=np.uint)
                self.setDataAt(a,y)
            else:
                self.__setHyperplane([x],y)
        elif isinstance(x, list):
            print ("List")
            raise Warning("To be implemented")
        elif isinstance(x, tuple):
            if len(x) is self.ndim:
                #TODO
                idx = np.zeros(len(x),dtype=np.uint)
                for ii in range(0,len(x)):
                    if(isinstance(x[ii],slice)):
                        self.setDataFromTuple(x, y)
                        return
                    else:
                        idx[ii] = x[ii]
                self.setDataAt(idx,y)
            elif len(x) > self.ndim:
                raise IndexError("too many indices for array")
            else:
                raise IndexError("too few indices for array")

        elif isinstance(x, np.ndarray):
            #TODO how efficient is this?? Probably a ptr would be better
            self.setDataAt(x,y)
        elif isinstance(x, slice):
            if self.ndim is 1:
                self.__idx_vec.resize(1)
                if isinstance(y,list) or isinstance(y,tuple) or isinstance(y,np.ndarray):
                    counter = 0
                    for ii in xrange(*(x).indices(self.shape[0])):
                        self.__setDataAt1D(ii,y[counter])
                        counter += 1
                else:
                    # for ii in xrange(*(x).indices(self.__shape_vec[0])):
                    for ii in xrange(self.__shape_vec[0]):
                        # self.__setDataAt1D(ii,y)
                        self.__idx_vec[0] = ii
                        # if(self.__idx_vec[0] >= self.__shape_vec[0]):
                        #     raise IndexError("index out of bound")
                        self.__setData(y)
                        pass

            else:
                raise Warning("To be implemented")
        else:
            print (type(x))
            raise TypeError("type unknown")

    def __len__(self):
        return self.shape[0]

    #########################
    cpdef toNpArray(self):
        #TODO CRITICAL
        res = np.zeros(self.shape,dtype=np.float)
        for index,value in np.ndenumerate(res):
            res[index] = self.getDataAt(np.asarray(index).astype(np.uint)) #TODO!!!! hprrible
        return res

    @property
    def shape(self):
        return tuple([self.__shape_vec[ii] for ii in xrange(0,self.__shape_vec.size())])

    @property
    def ndim(self):
        return self.__shape_vec.size()


cdef class View:

    cdef __getData(self): #uint64_t doesn't work here
        if self.__dtype == TypeId.FLOAT:
            return self.__typed_views.__cv_float.get()[0].dataAtSubSpace(self.__idx_vec)
        elif self.__dtype == TypeId.DOUBLE:
            return self.__typed_views.__cv_double.get()[0].dataAtSubSpace(self.__idx_vec)
        elif self.__dtype == TypeId.UINT32:
            return self.__typed_views.__cv_uint32.get()[0].dataAtSubSpace(self.__idx_vec)
        elif self.__dtype == TypeId.UINT64:
            return self.__typed_views.__cv_uint64.get()[0].dataAtSubSpace(self.__idx_vec)
        elif self.__dtype == TypeId.INT32:
            return self.__typed_views.__cv_int32.get()[0].dataAtSubSpace(self.__idx_vec)
        elif self.__dtype == TypeId.INT64:
            return self.__typed_views.__cv_int64.get()[0].dataAtSubSpace(self.__idx_vec)
        else:
            raise TypeError("type unknown")or("type unknown")

    # Type for y?
    cdef __setData(self, y): #uint64_t doesn't work here
        if self.__dtype == TypeId.FLOAT:
            self.__typed_views.__cv_float.get()[0].setDataAtSubSpace(self.__idx_vec,y)
        elif self.__dtype == TypeId.DOUBLE:
            self.__typed_views.__cv_double.get()[0].setDataAtSubSpace(self.__idx_vec,y)
        elif self.__dtype == TypeId.UINT32:
            self.__typed_views.__cv_uint32.get()[0].setDataAtSubSpace(self.__idx_vec,y)
        elif self.__dtype == TypeId.UINT64:
            self.__typed_views.__cv_uint64.get()[0].setDataAtSubSpace(self.__idx_vec,y)
        elif self.__dtype == TypeId.INT32:
            self.__typed_views.__cv_int32.get()[0].setDataAtSubSpace(self.__idx_vec,y)
        elif self.__dtype == TypeId.INT64:
            self.__typed_views.__cv_int64.get()[0].setDataAtSubSpace(self.__idx_vec,y)
        else:
            raise TypeError("type unknown")or("type unknown")

    cdef __getView(self): #uint64_t doesn't work here
        view = View()
        view.__dtype = self.__dtype
        view.__ind_idx = self.__ind_idx
        for i in range(view.__ind_idx.size()):
            if view.__ind_idx[i].size() > 1:
                view.__shape_vec.push_back(view.__ind_idx[i].size())
        if view.__shape_vec.empty():
            view.__shape_vec.push_back(1)

        if self.__dtype == TypeId.FLOAT:
            view.__typed_views.__cv_float = self.__typed_views.__cv_float.get()[0].getViewSubSpace(self.__ind_idx)
        elif self.__dtype == TypeId.DOUBLE:
            view.__typed_views.__cv_double = self.__typed_views.__cv_double.get()[0].getViewSubSpace(self.__ind_idx)
        elif self.__dtype == TypeId.UINT32:
            view.__typed_views.__cv_uint32 = self.__typed_views.__cv_uint32.get()[0].getViewSubSpace(self.__ind_idx)
        elif self.__dtype == TypeId.UINT64:
            view.__typed_views.__cv_uint64 = self.__typed_views.__cv_uint64.get()[0].getViewSubSpace(self.__ind_idx)
        elif self.__dtype == TypeId.INT32:
            view.__typed_views.__cv_int32 = self.__typed_views.__cv_int32.get()[0].getViewSubSpace(self.__ind_idx)
        elif self.__dtype == TypeId.INT64:
            view.__typed_views.__cv_int64 = self.__typed_views.__cv_int64.get()[0].getViewSubSpace(self.__ind_idx)
        else:
            raise TypeError("type unknown")or("type unknown")
        return view

    #########################
    cpdef getDataAt(self, unsigned long[:] idx):
        self.__idx_vec.resize(self.__shape_vec.size())
        for i in xrange(self.__shape_vec.size()):
            self.__idx_vec[i] = idx[i]
            if(self.__idx_vec[i] >= self.__shape_vec[i]):
                raise IndexError("index out of bound")
        return self.__getData()

    cpdef getViewFromTuple(self, x):
        view = View()
        self.__ind_idx.clear()
        self.__ind_idx.resize(self.ndim)
        for ii in range(0,len(x)):
            if(isinstance(x[ii],slice)):
                for jj in xrange(*(x[ii]).indices(self.shape[ii])):
                    self.__ind_idx[ii].push_back(jj)
            elif(isinstance(x[ii],int)):
                self.__ind_idx[ii].push_back(x[ii])
            else:
                raise TypeError("Unknown type in View creation")

        for ii in range(len(x),self.ndim):
            for jj in range(0,self.shape[ii]):
                self.__ind_idx[ii].push_back(jj) #iota

        return self.__getView()

    cpdef getViewFromInteger(self, x):
        view = View()
        self.__ind_idx.clear()
        self.__ind_idx.resize(self.ndim)
        self.__ind_idx[0].push_back(x)
        for ii in range(1,self.ndim):
            for jj in range(0,self.shape[ii]):
                self.__ind_idx[ii].push_back(jj) #iota

        return self.__getView()

    cpdef getViewFromSlice(self, x):
        view = View()
        self.__ind_idx.clear()
        self.__ind_idx.resize(self.ndim)
        for ii in xrange(*(x).indices(self.shape[0])):
            self.__ind_idx[0].push_back(ii)

        for jj in range(1,self.ndim):
            for ii in range(0,self.shape[jj]):
                self.__ind_idx[jj].push_back(ii) #iota

        return self.__getView()

    #########################
    def __getitem__(self,x):
        if len(self) is 0:
            return self
        #Integer
        if isinstance(x, int) or isinstance(x, np.int64):
            #1D case: direct access
            if self.ndim is 1:
                a = np.asarray([x],dtype=np.uint)
                return self.getDataAt(a)
            #else I return a view
            else:
                return self.getViewFromInteger(x)

        elif isinstance(x, list):
            if len(x) > self.ndim:
                raise IndexError("too many indices for array")
            elif len(x) is self.ndim:
                #direct indexing
                np_array = np.asarray(x,dtype=np.uint)
                return self.getDataAt(np_array)
            else:
                return self.getViewFromTuple(tuple(x))

        elif isinstance(x, tuple):
            if len(x) > self.ndim:
                raise IndexError("too many indices for array")
            #if the tuple does not contain any slicer, then return a values
            #otherwise returns a view
            elif len(x) is self.ndim:
                idx = np.zeros(len(x),dtype=np.uint)
                for ii in range(0,len(x)):
                    if(isinstance(x[ii],slice)):
                        return self.getViewFromTuple(x)
                    else:
                        idx[ii] = x[ii]
                return self.getDataAt(idx)
            else:
                return self.getViewFromTuple(x)

        #TODO how efficient is this??
        elif isinstance(x, np.ndarray):
            return self.getDataAt(x)

        elif isinstance(x, slice):
            return self.getViewFromSlice(x)

        else:
            print (type(x))
            raise TypeError("type unknown")

    #########################
    cpdef setDataAt(self, unsigned long[:] idx, float v):
        self.__idx_vec.resize(self.__shape_vec.size())
        for i in xrange(self.__shape_vec.size()):
            #TODO check efficiency
            self.__idx_vec[i] = idx[i]
            if(self.__idx_vec[i] >= self.__shape_vec[i]):
                raise IndexError("index out of bound")
        self.__setData(v)

    #########################
    def __setitem__(self,x,y):
        #TODO implement here the type identification
        # print(self.__shape_vec)
        #print(x)
        if isinstance(x, int) or isinstance(x, np.int64):
            print ("Integer")
            raise Warning("To be implemented")
        elif isinstance(x, list):
            print ("List")
            raise Warning("To be implemented")
        elif isinstance(x, tuple):
            print ("Tuple")
            raise Warning("To be implemented")
        elif isinstance(x, np.ndarray):
            #TODO how efficient is this?? Probably a ptr would be better
            self.setDataAt(x,y)
        elif isinstance(x, slice):
            if self.ndim is 1:
                if isinstance(y,list) or isinstance(y,tuple) or isinstance(y,np.ndarray):
                    counter = 0
                    for ii in xrange(*(x).indices(self.shape[0])):
                        a = np.asarray([ii],dtype=np.uint) ##### HORRIBLE
                        self.setDataAt(a,y[counter])
                        counter += 1
                else:
                    for ii in xrange(*(x).indices(self.shape[0])):
                        a = np.asarray([ii],dtype=np.uint) ##### HORRIBLE
                        self.setDataAt(a,y)
            else:
                raise Warning("To be implemented")
        else:
            print (type(x))
            raise TypeError("type unknown")

    @property
    def shape(self):
        return tuple([self.__shape_vec[ii] for ii in xrange(0,self.__shape_vec.size())])

    @property
    def ndim(self):
        return self.__shape_vec.size()

    def __len__(self):
        return self.shape[0]

    cpdef toNpArray(self):
        #TODO CRITICAL
        res = np.zeros(self.shape,dtype=np.float)
        for index,value in np.ndenumerate(res):
            res[index] = self.getDataAt(np.asarray(index).astype(np.uint)) #TODO!!!! hprrible
        return res

    def tolist(self):
        #TODO CRITICAL
        return self.toNpArray().tolist()
