/*
 *
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#ifndef CHUNKED_MEMORY_INL
#define CHUNKED_MEMORY_INL

#include "pmm/chunked_memory.h"
#include <memory>
#include <iostream>
#include <omp.h>

namespace pmm{

    template <typename value_type>
    ChunkedMemory<value_type>::ChunkedMemory():
    _memory_policy(nullptr){
    }

    template <typename value_type>
    ChunkedMemory<value_type>::ChunkedMemory(const shape_type& shape, const shape_type& chunk_shape):
        _memory_policy(nullptr)
    {
        setShape(shape,chunk_shape);
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::setShape(const shape_type& shape, const shape_type& chunk_shape){
        _shape = shape;
        _num_dims = shape.size();
        _chunk_shape = chunk_shape;
        assert(_shape.size() == _chunk_shape.size());
        for(int i = 0; i < _num_dims; ++i){
            if(chunk_shape[i] == 0){
                throw std::logic_error("chunk dimensions must be bigger than 0");
            }
        }
        //Initialization of the 1D case
        if(_num_dims == 1){
           _chunk_shape_1D = chunk_shape[0];
           _temp_cached_chunk_id = std::numeric_limits<uint64_t>::max();
           _temp_cached_chunk_ptr = nullptr;
        }

        sub_grid_type sub_grid;

        _chunk_grid_shape.resize(_shape.size());
        sub_grid._chunk_grid_extension.resize(_shape.size());
        sub_grid._chunk_grid_shape.resize(_shape.size());
        sub_grid._chunk_grid_incremental_shape.resize(_shape.size());
        sub_grid._num_chunks = 1;
        sub_grid._start_lin_idx = 0;
        _chunk_size = 1;

        _length = 1;
        for(int i = 0; i < _num_dims; ++i){
            _length *= _shape[i];
            _chunk_grid_shape[i] = std::ceil(float(_shape[i])/_chunk_shape[i]);
            sub_grid._chunk_grid_extension[i] = _chunk_grid_shape[i];
            sub_grid._chunk_grid_shape[i] = _chunk_grid_shape[i];
            sub_grid._chunk_grid_incremental_shape[i] = _chunk_grid_shape[i];
            sub_grid._num_chunks *= _chunk_grid_shape[i];
            _chunk_size *= _chunk_shape[i];
        }


        initializeCache();
        initializeStrides();

        sub_grid._strides_grid = _strides_grid;
        _sub_grids.push_back(sub_grid);
        _num_sub_grids = _sub_grids.size();
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::reshape(const shape_type& shape){
        _length = 1;
        //if it is increasing in size I add a sub grid for every dimension
        for(int i = 0; i < _num_dims; ++i){
            if(_chunk_grid_shape[i]*_chunk_shape[i] < shape[i]){
                addSubGrid(shape,i);
            }
            _length *= shape[i];
        }
        _shape = shape;
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::addSubGrid(const shape_type& shape, uint32_t dim){
        auto& prev_grid = _sub_grids[_sub_grids.size()-1];
        sub_grid_type sub_grid;
        //computing the shape of the subgrid
        sub_grid._chunk_grid_extension = shape_type(_num_dims,0);
        sub_grid._chunk_grid_extension[dim] = std::ceil(float(shape[dim])/_chunk_shape[dim]);
        sub_grid._chunk_grid_extension[dim] -= _chunk_grid_shape[dim];

        sub_grid._chunk_grid_shape = _chunk_grid_shape;
        sub_grid._chunk_grid_shape[dim] = sub_grid._chunk_grid_extension[dim];
        sub_grid._strides_grid.resize(_num_dims);
        sub_grid._start_lin_idx = prev_grid._num_chunks+prev_grid._start_lin_idx;

        sub_grid._num_chunks = 1;
        for(int i = 0; i < _num_dims; ++i){
            sub_grid._num_chunks *= sub_grid._chunk_grid_shape[i];
        }

        //sub grid strides
        uint64_t grid_stride = 1;
        for(int i = _num_dims-1; i >=0 ; --i){
            sub_grid._strides_grid[i] = grid_stride;
            grid_stride *= sub_grid._chunk_grid_shape[i];
        }

        _chunk_grid_shape[dim] += sub_grid._chunk_grid_extension[dim];
       sub_grid._chunk_grid_incremental_shape = _chunk_grid_shape;

        _sub_grids.push_back(sub_grid);
        _num_sub_grids = _sub_grids.size();
    }

    template <typename value_type>
    value_type& ChunkedMemory<value_type>::operator[](const idx_type& idx){
        assert(idx.size() == _num_dims);
        uint32_t cache_slot = omp_get_thread_num();
        auto& cache = _cache[cache_slot];
        //if(_sub_grids.size() == 1){
        if(_num_sub_grids == 1){
            cacheIndices(idx,cache_slot);
        }else{
            cacheIndicesWithSubGrids(idx,cache_slot);
        }

        chunk_linear_idx_type chunk_lin_idx(cache._chunk_lin_idx);
        auto it = _loaded_chunks.find(chunk_lin_idx);
        if(it == _loaded_chunks.end()){
            return (loadInMemory(chunk_lin_idx))->ptr()[cache._idx_in_chunk];
        }else{
            return (*(it->second))[cache._idx_in_chunk];
        }
    }

    template <typename value_type>
    value_type& ChunkedMemory<value_type>::operator[](uint64_t idx){
        assert(_num_dims == 1);
        uint64_t chunk_lin_idx = idx/_chunk_shape_1D;
        uint64_t idx_in_chunk  = idx%_chunk_shape_1D;

        //if(_sub_grids.size() == 1){
        if(_num_sub_grids == 1){
//            cacheIndices1D(idx,cache_slot);
        }else{
            assert(false);
            //cacheIndicesWithSubGrids(idx,cache_slot);
        }

        if(chunk_lin_idx == _temp_cached_chunk_id){
            //assert(_temp_cached_chunk_ptr);
            return (*_temp_cached_chunk_ptr)[idx_in_chunk];
        }else{
            _temp_cached_chunk_id = chunk_lin_idx;
            auto it = _loaded_chunks.find(chunk_lin_idx);
            if(it == _loaded_chunks.end()){
                _temp_cached_chunk_ptr = loadInMemory(chunk_lin_idx);
            }else{
                _temp_cached_chunk_ptr = it->second.get();
            }
            //assert(_temp_cached_chunk_ptr);
            return (*_temp_cached_chunk_ptr)[idx_in_chunk];
        }
    }



    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_type* ChunkedMemory<value_type>::loadInMemory(const idx_type& idx){
        uint32_t cache_slot = omp_get_thread_num();
        auto& cache = _cache[cache_slot];
        idxToMatChunkIdx(idx,cache_slot);
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        return loadInMemory(cache._chunk_lin_idx);
    }

    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_type* ChunkedMemory<value_type>::loadInMemory(const idx_type& idx, uint32_t cache_slot){
        idxToMatChunkIdx(idx,cache_slot);
        auto& cache = _cache[cache_slot];
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        return loadInMemory(cache._chunk_lin_idx);
    }

    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_type* ChunkedMemory<value_type>::loadInMemory(chunk_linear_idx_type chunk_lin_idx){
        //Chunk already in memory
        assert(!_chunks_in_memory.contains(chunk_lin_idx));

        if(_memory_policy!=nullptr){_memory_policy->onBeforeChunkLoad(chunk_lin_idx);}

        //Chunk creation
        _loaded_chunks[chunk_lin_idx] = std::unique_ptr<chunk_type>(new chunk_type(_chunk_size));
        auto chunk = _loaded_chunks[chunk_lin_idx].get(); //TOOPTIMIZE

        //Bitmaps are updated
        //TODO ROARING BITMAPS ARE 32 BITS!
        _chunks_in_memory.add(chunk_lin_idx);
        _chunks_initialized.add(chunk_lin_idx);

        if(_memory_policy!=nullptr){_memory_policy->onAfterChunkLoad(chunk_lin_idx,chunk);}
        return chunk;
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::removeFromMemory(const idx_type& idx){
        uint32_t cache_slot = omp_get_thread_num();
        auto& cache = _cache[cache_slot];
        idxToMatChunkIdx(idx,cache_slot);
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        removeFromMemory(cache._chunk_lin_idx);
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::removeFromMemory(const idx_type& idx, uint32_t cache_slot){
        idxToMatChunkIdx(idx,cache_slot);
        auto& cache = _cache[cache_slot];
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        removeFromMemory(cache._chunk_lin_idx);
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::removeFromMemory(chunk_linear_idx_type chunk_lin_idx){
        //Chunk must be in memory
        assert(_chunks_in_memory.contains(chunk_lin_idx));

        //Chunk creation
        auto it = _loaded_chunks.find(chunk_lin_idx);
        _loaded_chunks.erase(it);

        //Bitmaps are updated
        //TODO ROARING BITMAPS ARE 32 BITS!
        _chunks_in_memory.remove(chunk_lin_idx);
        _chunks_initialized.remove(chunk_lin_idx);
    }

    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_type* ChunkedMemory<value_type>::getChunk(const idx_type& idx){
        uint32_t cache_slot = omp_get_thread_num();
        auto& cache = _cache[cache_slot];
        idxToMatChunkIdx(idx,cache_slot);
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        return getChunk(cache._chunk_lin_idx);
    }

    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_type* ChunkedMemory<value_type>::getChunk(const idx_type& idx, uint32_t cache_slot){
        idxToMatChunkIdx(idx,cache_slot);
        auto& cache = _cache[cache_slot];
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        return getChunk(cache._chunk_lin_idx);
    }

    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_type* ChunkedMemory<value_type>::getChunk(chunk_linear_idx_type chunk_lin_idx){
        //Chunk already in memory
        assert(_chunks_in_memory.contains(chunk_lin_idx));
        return _loaded_chunks[chunk_lin_idx].get();
    }


    template <typename value_type>
    bool ChunkedMemory<value_type>::inMemory(const idx_type& idx)const{
        uint32_t cache_slot = omp_get_thread_num();
        auto& cache = _cache[cache_slot];
        idxToMatChunkIdx(idx,cache_slot);
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        return _chunks_in_memory.contains(cache._chunk_lin_idx);
    }
    template <typename value_type>
    bool ChunkedMemory<value_type>::inMemory(chunk_linear_idx_type chunk_lin_idx)const{
        return _chunks_in_memory.contains(chunk_lin_idx);
    }

    template <typename value_type>
    bool ChunkedMemory<value_type>::initialized(const idx_type& idx)const{
        uint32_t cache_slot = omp_get_thread_num();
        auto& cache = _cache[cache_slot];
        idxToMatChunkIdx(idx,cache_slot);
        matToLinearChunkIdx(cache._chunk_mat_idx,cache_slot);
        return _chunks_initialized.contains(cache._chunk_lin_idx);
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::cacheIndicesWithSubGrids(const idx_type& idx, uint32_t cache_slot)const{
        auto& cache = _cache[cache_slot];
        auto& chunk_mat_idx = cache._chunk_mat_idx;
        auto& chunk_lin_idx = cache._chunk_lin_idx;
        auto& idx_in_chunk  = cache._idx_in_chunk;

        for(int i = 0; i < _num_dims; ++i){
            chunk_mat_idx[i] = idx[i]/_chunk_shape[i];
        }
        int sub_grid_id = 0;
        for(; sub_grid_id < _sub_grids.size()-1; ++sub_grid_id){
            bool valid = true;
            for(int i = 0; i < _num_dims && valid; ++i){
                if(chunk_mat_idx[i] >= _sub_grids[sub_grid_id]._chunk_grid_incremental_shape[i]){
                    valid = false;
                }
            }
            if(valid){
                break;
            }
        }

        chunk_lin_idx = _sub_grids[sub_grid_id]._start_lin_idx;
        idx_in_chunk = 0;
        for(int i = 0; i < _num_dims; ++i){
            idx_in_chunk  += (idx[i]-chunk_mat_idx[i]*_chunk_shape[i])*_strides_chunk[i];
            chunk_mat_idx[i] -= (_sub_grids[sub_grid_id]._chunk_grid_incremental_shape[i] - _sub_grids[sub_grid_id]._chunk_grid_shape[i]);
            chunk_lin_idx += chunk_mat_idx[i]*_sub_grids[sub_grid_id]._strides_grid[i];
        }
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::cacheIndices(const idx_type& idx, uint32_t cache_slot)const{
        auto& cache = _cache[cache_slot];
        auto& chunk_mat_idx = cache._chunk_mat_idx;
        auto& chunk_lin_idx = cache._chunk_lin_idx;
        auto& idx_in_chunk  = cache._idx_in_chunk;

        chunk_lin_idx = 0;
        idx_in_chunk = 0;
        for(int i = 0; i < _num_dims; ++i){
            chunk_mat_idx[i] = idx[i]/_chunk_shape[i];
            chunk_lin_idx += chunk_mat_idx[i]*_strides_grid[i];
            idx_in_chunk  += (idx[i]-chunk_mat_idx[i]*_chunk_shape[i])*_strides_chunk[i];
        }
    }

    //not really needed
    template <typename value_type>
    void ChunkedMemory<value_type>::cacheIndices1D(uint64_t idx, uint32_t cache_slot)const{
        auto& cache = _cache[cache_slot];
        auto& chunk_lin_idx = cache._chunk_lin_idx;
        auto& idx_in_chunk  = cache._idx_in_chunk;
        chunk_lin_idx = idx/_chunk_shape_1D;
        idx_in_chunk  = idx%_chunk_shape_1D;
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::idxToMatChunkIdx(const idx_type& idx, uint32_t cache_slot)const{
        auto& chunk_mat_idx = _cache[cache_slot]._chunk_mat_idx;
        for(int i = 0; i < _num_dims; ++i){
            chunk_mat_idx[i] = idx[i]/_chunk_shape[i];
        }
    }
    template <typename value_type>
    void ChunkedMemory<value_type>::matToLinearChunkIdx(const chunk_mat_idx_type& mat_idx, uint32_t cache_slot)const{
        chunk_linear_idx_type res = 0;
        uint64_t multiplier = 1;
        for(int i = _num_dims-1; i >= 0 ; --i){
            res += mat_idx[i]*multiplier;
            multiplier *= _chunk_grid_shape[i];
        }
        _cache[cache_slot]._chunk_lin_idx = res;
    }
    template <typename value_type>
    void ChunkedMemory<value_type>::linearToMatChunkIdx(const chunk_linear_idx_type lin_idx, uint32_t cache_slot)const{
        auto& chunk_mat_idx = _cache[cache_slot]._chunk_mat_idx;
        uint64_t residual = lin_idx;
        for(int i = _num_dims-1; i >= 0 ; --i){
            chunk_mat_idx[i] = residual%_chunk_grid_shape[i];
            residual /= _chunk_grid_shape[i];
        }
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::idxInChunk(const idx_type& idx, const chunk_mat_idx_type& chunk_idx, uint32_t cache_slot)const{
        auto& cache = _cache[cache_slot];
        cache._idx_in_chunk = 0;
        uint64_t multiplier = 1;
        for(int i = _num_dims-1; i >= 0 ; --i){
            cache._idx_in_chunk += (idx[i]-chunk_idx[i]*_chunk_shape[i])*multiplier;
            multiplier *= _chunk_shape[i];
        }
    }

/////////////////////////////////////////////////////////////////////////
    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_mat_idx_type ChunkedMemory<value_type>::idxToMatChunkIdx(const idx_type& idx)const{
        chunk_mat_idx_type res(_num_dims);
        for(int i = 0; i < _num_dims; ++i){
            res[i] = idx[i]/_chunk_shape[i];
        }
        return res;
    }
    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_linear_idx_type ChunkedMemory<value_type>::matToLinearChunkIdx(const chunk_mat_idx_type& mat_idx)const{
        chunk_linear_idx_type res = 0;
        uint64_t multiplier = 1;
        for(int i = _num_dims-1; i >= 0 ; --i){
            res += mat_idx[i]*multiplier;
            multiplier *= _chunk_grid_shape[i];
        }
        return res;
    }
    template <typename value_type>
    typename ChunkedMemory<value_type>::chunk_mat_idx_type ChunkedMemory<value_type>::linearToMatChunkIdx(const chunk_linear_idx_type lin_idx)const{
        chunk_mat_idx_type res(_num_dims);
        uint64_t residual = lin_idx;
        for(int i = _num_dims-1; i >= 0 ; --i){
            res[i] = residual%_chunk_grid_shape[i];
            residual /= _chunk_grid_shape[i];
        }
        return res;
    }

    template <typename value_type>
    uint64_t ChunkedMemory<value_type>::idxInChunk(const idx_type& idx, const chunk_mat_idx_type& chunk_idx)const{
        idx_type residual(_num_dims);
        for(int i = 0; i < _num_dims; ++i){
            residual[i] = idx[i]-chunk_idx[i]*_chunk_shape[i];
        }

        chunk_linear_idx_type res = 0;
        uint64_t multiplier = 1;
        for(int i = _num_dims-1; i >= 0 ; --i){
            res += residual[i]*multiplier;
            multiplier *= _chunk_shape[i];
        }
        return res;
    }

/////////////////////////////////////////////////////////////////////////

    template <typename value_type>
    std::shared_ptr<typename ChunkedMemory<value_type>::view_type> ChunkedMemory<value_type>::getView(const idx_ind_type& idx_ind){
        auto ptr = std::make_shared<view_type>(idx_ind);
        ptr->_memory_ptr = this;
        return ptr;
    }

/////////////////////////////////////////////////////////////////////////

    template <typename value_type>
    void ChunkedMemory<value_type>::initializeCache(){
        int num_threads = omp_get_max_threads();
        //std::cout << "#threads:\t" << num_threads << std::endl;
        _cache.resize(num_threads);
        for(int c = 0; c < _cache.size(); ++c){
            _cache[c]._data_idx.resize(_num_dims);
            _cache[c]._chunk_mat_idx.resize(_num_dims);
        }
    }

    template <typename value_type>
    void ChunkedMemory<value_type>::initializeStrides(){
        _strides_grid.resize(_num_dims);
        _strides_chunk.resize(_num_dims);
        uint64_t grid_stride = 1;
        uint64_t chunk_stride = 1;
        for(int i = _num_dims-1; i >=0 ; --i){
            _strides_grid[i]    = grid_stride;
            _strides_chunk[i]   = chunk_stride;
            grid_stride         *= _chunk_grid_shape[i];
            chunk_stride        *= _chunk_shape[i];
        }
    }
}

#endif
