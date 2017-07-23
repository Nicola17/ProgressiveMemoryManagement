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

#ifndef POLICY_R_BIN_FILE_H
#define POLICY_R_BIN_FILE_H

#include <stdint.h>
#include <cassert>
#include <string>
#include <queue>
#include "abstract_memory_policy.h"

namespace pmm{

    //! Read from file memory policy
    /*!
        Read chuncks from a binary file and keeps a limited number of chunks in memory
        \author Nicola Pezzotti
    */
    template <typename value_type>
    class ReadFromFileBinaryPolicy: public AbstractMemoryPolicy<value_type>{
    public:
        typedef Chunk<value_type> chunk_type;

    public:
        ReadFromFileBinaryPolicy(std::string filename, uint64_t max_loaded_chunks = 100):
            _filename(filename),
            _max_loaded_chunks(max_loaded_chunks),
            _tick(0)
        {

        }
        virtual ~ReadFromFileBinaryPolicy(){}

        virtual void initialize(ChunkedMemory<value_type>* mem){
            _mem = mem;
            _shape = mem->shape();
            _chunk_shape = mem->chunk_shape();
            if(_shape.size() != 2){
                throw std::logic_error("ReadFromFileBinaryPolicy works for 2D arrays");
            }
            if(_shape[1] != _chunk_shape[1]){
                throw std::logic_error("ReadFromFileBinaryPolicy: shape[1] != chunk_shape[1]");
            }
        }
        virtual void onBeforeChunkLoad(uint64_t idx){}
        virtual void onAfterChunkLoad(uint64_t idx, chunk_type* ptr){
            //Here we assume a linear chunk space
            std::ifstream f(_filename.c_str(),std::ios::binary);
            uint64_t offset = idx*ptr->size()*sizeof(value_type);
            f.seekg (offset, f.beg);
            f.read((char*)ptr->ptr(),ptr->size()*sizeof(value_type));
            value_type v0 = ptr->ptr()[0];
            value_type v1 = ptr->ptr()[1];
            value_type v2 = ptr->ptr()[2];

            _history.push(std::pair<uint64_t,uint64_t>(_tick,idx));
            ++_tick; //TODO overflow
            //if the more chuncks are in memory than _max_loaded_chucks I remove the oldest one
            if(_history.size() > _max_loaded_chunks){
                auto idx_to_remove = std::get<1>(_history.top());
                //remove from the priority queue
                _history.pop();
                //remove chunk from memory
                _mem->removeFromMemory(idx_to_remove);
            }
        }

    private:
        std::string _filename;
        uint64_t    _max_loaded_chunks;
        ChunkedMemory<value_type>* _mem;
        typename ChunkedMemory<value_type>::shape_type _shape;
        typename ChunkedMemory<value_type>::shape_type _chunk_shape;
        typedef std::pair<uint64_t,uint64_t> history_value_type;
        std::priority_queue<history_value_type,std::vector<history_value_type>,std::greater<history_value_type>> _history;
        uint64_t    _tick;
    };

}


#endif
