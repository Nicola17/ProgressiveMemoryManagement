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

#ifndef SGD_BIN_FILE_INL
#define SGD_BIN_FILE_INL

#include "pmm/sgd_bin_file.h"
#include <memory>
#include <iostream>
#include <cassert>
#include <omp.h>

namespace pmm{


    template <typename value_type>
    SGDBinFile<value_type>::SGDBinFile(const std::string& str, const shape_type& shape):
        _shape(shape)
    {
        assert(shape.size() == 2);
        const int MAX_CHUNK_SHAPE = 4*1024;
        _chunk_shape = shape;
        _chunk_shape[0] = MAX_CHUNK_SHAPE / _chunk_shape[1];
        _memory.setShape(_shape,_chunk_shape);
        std::cout << "Chunk shape: (" << _chunk_shape[0] << "," << _chunk_shape[1] << ")" << std::endl;


        //smart sample
        _max_chunks_in_memory = 100;
        _samples_per_chunk = _chunk_shape[0]*0.2;
        _n_sampled = 0;

        //Initializing the random number generator;
        _distribution_pnts      = std::uniform_int_distribution<>(0,shape[0]-1);
        _distribution_chunks    = std::uniform_int_distribution<>(0,_max_chunks_in_memory-1);
        _distribution_in_chunks = std::uniform_int_distribution<>(0,_chunk_shape[0]-1);

        //File
        _in_file.open(str.c_str(), std::ios::in | std::ios::binary);
        if(!_in_file.is_open()){
            throw std::runtime_error("Couldn't open the input file");
        }
    }

    template <typename value_type>
    value_type* SGDBinFile<value_type>::getRandomData(){
        if(_loaded_chunks.size() < _max_chunks_in_memory){
            //std::cout << "Loading a new chunk..." << std::endl;
            uint64_t idx = getRandomIdx();
            uint64_t idx_chunk = idx/_chunk_shape[0];
            while(_memory.inMemory(idx_chunk)){
                idx_chunk = (++idx_chunk); //TODO%()
            }

            auto chunk_ptr = _memory.loadInMemory(idx_chunk);
            _loaded_chunks.push_back(chunk_ptr);
            _loaded_chunks_idx.push_back(idx_chunk);

            //Load from file
            _in_file.seekg(idx*_chunk_shape[1]*sizeof(float),_in_file.beg);
            _in_file.read((char*)chunk_ptr->ptr(), sizeof(value_type)*_chunk_shape[0]*_chunk_shape[1]);

            auto pnt_in_chunk = 0;//_distribution_in_chunks(_generator);
            return &(chunk_ptr->ptr()[pnt_in_chunk*_chunk_shape[1]]);
        }

        if(_n_sampled >= _samples_per_chunk){
            _n_sampled = 0;
            _loaded_chunks.pop_front();
            auto idx = _loaded_chunks_idx.front();
            _loaded_chunks_idx.pop_front();

            //remove from memory
            _memory.removeFromMemory(idx);
        }

        auto chunk_in_queue = rand()%_loaded_chunks_idx.size();//_distribution_chunks(_generator);
        auto pnt_in_chunk = _distribution_in_chunks(_generator);
        ++_n_sampled;
        return &(_loaded_chunks[chunk_in_queue]->ptr()[pnt_in_chunk*_chunk_shape[1]]);
        //return _loaded_chunks[chunk_in_queue]->ptr();
    }

    template <typename value_type>
    void SGDBinFile<value_type>::getMultipleDataPoints(std::vector<value_type>& pnts, unsigned int n){
        auto n_dim = _shape[1];
        pnts.resize(n*n_dim);
        for(int i = 0; i < n; ++i){
            auto ptr = getRandomData();
            for(int d = 0; d < n_dim; ++d){
                pnts[i*n_dim+d] = ptr[d];
            }
        }
    }

    template <typename value_type>
    uint64_t SGDBinFile<value_type>::getRandomIdx(){
        return _distribution_pnts(_generator);
    }


}

#endif
