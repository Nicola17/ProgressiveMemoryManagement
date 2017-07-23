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

#ifndef CHUNKED_MEMORY_H
#define CHUNKED_MEMORY_H

#include <stdint.h>
#include "pmm/chunk.h"
#include "pmm/view.h"
#include "pmm/roaring.hh"
#include "pmm/abstract_memory_policy.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>

namespace pmm{

    template <typename value_type>
    class View;

    //! Chunked memory
    /*!
        \author Nicola Pezzotti
    */
    template <typename value_type>
    class ChunkedMemory{
    public:
        enum ChunkState {UNINITIALIZED = 0, IN_MEMORY = 1, IN_GPU_MEMORY = 2, ON_DISK = 4, ON_DIFF_NODE = 8};
    public:
        //TODO clean types
        typedef Roaring                     bitmap_type;
        typedef std::vector<uint64_t>       shape_type;
        typedef std::vector<uint64_t>       idx_type;
        typedef std::vector<uint64_t>       chunk_mat_idx_type;
        typedef uint64_t                    chunk_linear_idx_type;
        typedef uint32_t                    chunk_states_type;

        typedef std::vector<uint64_t> uint64_vec_type;
        typedef std::vector<uint64_vec_type> idx_ind_type;
        typedef View<value_type>            view_type;


        typedef Chunk<value_type>                           chunk_type;
        //TODO unique pointer is probably not a good idea...
        typedef std::unordered_map<uint64_t,std::unique_ptr<chunk_type>>     chunk_storage_type;
        typedef AbstractMemoryPolicy<value_type> policy_type;

    private:
        typedef struct{
            idx_type                _data_idx;
            chunk_mat_idx_type      _chunk_mat_idx;
            chunk_linear_idx_type   _chunk_lin_idx;
            chunk_linear_idx_type   _idx_in_chunk;
        } indexes_cache_type;
        typedef std::vector<indexes_cache_type> cache_type;

        //! data structure used to dynamically change the container
        typedef struct{
            shape_type _shape;
            shape_type _chunk_grid_extension; //TODO can be a pair if the initial grid size is handled separatedly
            shape_type _chunk_grid_shape;
            shape_type _chunk_grid_incremental_shape;
            shape_type _strides_grid;
            uint64_t   _num_chunks;
            uint64_t   _start_lin_idx;
        } sub_grid_type;
        typedef std::vector<sub_grid_type> sub_grids_type;


    public:
        ChunkedMemory();
        ChunkedMemory(const shape_type& shape, const shape_type& chunk_shape);

        //setter and getter for shapes
        void setShape(const shape_type& shape, const shape_type& chunk_shape);
        //reshape the container (not the chunks)
        void reshape(const shape_type& shape);

        value_type& operator[](const idx_type& idx);
        value_type& operator[](uint64_t idx);

        std::shared_ptr<view_type> getView(const idx_ind_type& idx_ind);

        uint64_t size(uint64_t d)               const{assert(d < _shape.size()); return _shape[d];}
        const shape_type& shape()               const{return _shape;}
        const shape_type& chunk_shape()         const{return _chunk_shape;}
        const shape_type& chunk_grid_shape()    const{return _chunk_grid_shape;}
        uint64_t          chunk_size()          const{return _chunk_size;}
        uint64_t          chunk_byte_size()     const{return _chunk_size*sizeof(value_type);}
        uint64_t          length()              const{return _length;}
        uint64_t          ndim()                const{return _num_dims;}

        uint64_t          loadedChunks()        const{return _loaded_chunks.size();}

        policy_type* get_memory_policy(){return _memory_policy;}
        void set_memory_policy(policy_type* mp){_memory_policy = mp; _memory_policy->initialize(this);}//some checks must be done here if the mem was already used


/////////////////////////////////////////////////


        //TOREMOVE -> just for thest
        void test1(const idx_type& idx){
          std::cout << idx.size() << std::endl;
          for(const auto v: idx) std::cout << v << " " << std::endl;
          std::cout << "Cache: " << _cache.size() << std::endl;
          std::cout << "Shape: " << _shape.size() << std::endl;
          for(const auto v: _shape) std::cout << v << " " << std::endl;
          std::cout << "Chunk Shape: " << _chunk_shape.size() << std::endl;
          for(const auto v: _chunk_shape) std::cout << v << " " << std::endl;
         }
        //value_type& test1(const idx_type& idx);




        //!Optimizes the bijective mapping between the linear and matricial chunck indexing
        void optimizeChunks(){std::cout << "TO BE IMPLEMENTED\n";}

        bool inMemory   (const idx_type& idx)const;
        bool inMemory   (chunk_linear_idx_type idx)const;
        bool initialized(const idx_type& idx)const;

        chunk_type* loadInMemory(const idx_type& idx);
        chunk_type* loadInMemory(const idx_type& idx, uint32_t cache_slot);
        chunk_type* loadInMemory(chunk_linear_idx_type idx);

        void removeFromMemory(const idx_type& idx);
        void removeFromMemory(const idx_type& idx, uint32_t cache_slot);
        void removeFromMemory(chunk_linear_idx_type idx);

        chunk_type* getChunk(const idx_type& idx);
        chunk_type* getChunk(const idx_type& idx, uint32_t cache_slot);
        chunk_type* getChunk(chunk_linear_idx_type idx);

        //const value_type& operator[](idx_type idx) const;
        void idxToMatChunkIdx(const idx_type& idx, uint32_t cache_slot)const;
        void matToLinearChunkIdx(const chunk_mat_idx_type& mat_idx, uint32_t cache_slot)const;
        void linearToMatChunkIdx(const chunk_linear_idx_type lin_idx, uint32_t cache_slot)const;
        void idxInChunk(const idx_type& idx, const chunk_mat_idx_type& chunk_idx, uint32_t cache_slot)const;


        //! NB: slow - computes the ch
        chunk_mat_idx_type idxToMatChunkIdx(const idx_type& idx)const;
        chunk_linear_idx_type matToLinearChunkIdx(const chunk_mat_idx_type& mat_idx)const;
        chunk_mat_idx_type linearToMatChunkIdx(const chunk_linear_idx_type lin_idx)const;
        uint64_t idxInChunk(const idx_type& idx, const chunk_mat_idx_type& chunk_idx)const;

    private:
        void cacheIndices(const idx_type& idx, uint32_t cache_slot)const;
        void cacheIndices1D(uint64_t idx, uint32_t cache_slot)const;
        void cacheIndicesWithSubGrids(const idx_type& idx, uint32_t cache_slot)const;

        void initializeCache();
        void initializeStrides();
        void addSubGrid(const shape_type& shape, uint32_t dim);

    private:
        shape_type _shape;
        shape_type _chunk_grid_shape;
        shape_type _strides_grid;
        chunk_storage_type _loaded_chunks;

        //for reshapeable data
        sub_grids_type _sub_grids;


        shape_type _chunk_shape;
        shape_type _strides_chunk;
        uint64_t   _chunk_size;
        uint32_t   _num_dims;
        uint32_t   _num_sub_grids;
        uint64_t   _length;

        //1Dimensional
        uint64_t _chunk_shape_1D;
        uint64_t _temp_cached_chunk_id;
        chunk_type* _temp_cached_chunk_ptr;


        bitmap_type _chunks_initialized;
        bitmap_type _chunks_in_memory;
        bitmap_type _chunks_in_gpu_memory;
        bitmap_type _chunks_on_disk;
        bitmap_type _chunks_on_diff_node;

        policy_type* _memory_policy;

        mutable cache_type _cache;
    };

}

#endif
