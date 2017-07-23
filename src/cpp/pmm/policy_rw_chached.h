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

#ifndef POLICY_RW_CACHED_H
#define POLICY_RW_CACHED_H

#include <stdint.h>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <queue>
#include <tuple>
#include "abstract_memory_policy.h"
#include "pmm/chunked_memory.h"

namespace pmm{

    //! Read/Write memory policy with cache on disk.
    /*!
        Read/Write memory policy with cache on disk.
        \author Nicola Pezzotti
        \warning not working ATM
        \note switch to HDF5 maybe?
    */
    template <typename value_type>
    class ReadWriteCachedPolicy: public AbstractMemoryPolicy<value_type>{
    public:
        typedef Chunk<value_type> chunk_type;

    public:
        ReadWriteCachedPolicy(std::string directory, uint64_t max_loaded_chunks = 100, value_type init_val = 0):
            _init_val(init_val),
            _directory(directory),
            _tick(0),
            _max_loaded_chunks(max_loaded_chunks),
            _mem(nullptr)
        {
            _id = 0; //TODO

        }
        virtual ~ReadWriteCachedPolicy(){}

        virtual void initialize(ChunkedMemory<value_type>* mem){ _mem = mem;}

        virtual void onBeforeChunkLoad(uint64_t idx){}
        virtual void onAfterChunkLoad(uint64_t idx, chunk_type* ptr){
            assert(_mem != nullptr);
            std::stringstream ss;
            ss << _directory << "pmm_" << _id << "_chunk_" << idx << ".bin";
            std::ifstream f(ss.str().c_str(),std::ios::binary);
            if(f.fail()){
                //if it is not cached on disk I initialize the memory with the proper init value
                std::fill_n(ptr->ptr(),ptr->size(),_init_val);
                _history.push(std::pair<uint64_t,uint64_t>(_tick,idx));
                ++_tick; //overflow? TODO
            }else{
                //if it is cached I load the value from disk
                loadChunkFromDisk(ptr,f,ss.str());
            }

            //if the more chuncks are in memory than _max_loaded_chucks I remove the oldest one
            if(_history.size() > _max_loaded_chunks){
                auto idx_to_remove = std::get<1>(_history.top());
                std::stringstream ss;
                ss << _directory << "pmm_" << _id << "_chunk_" << idx_to_remove << ".bin";

                //cache the data on disk
                std::ofstream of(ss.str().c_str(), std::ios::binary);
                if(of.fail()){
                    throw std::runtime_error("unable to cache data on disk!");
                }
                auto chunk_to_remove = _mem->getChunk(idx_to_remove);
                of.write((char*)(chunk_to_remove->ptr()), chunk_to_remove->size()*sizeof(value_type));

                //remove from the priority queue
                _history.pop();

                //remove chunk from memory
                _mem->removeFromMemory(idx_to_remove);
            }
        }

        void saveChunkOnDisk(){

        }
        void loadChunkFromDisk(chunk_type* chunk, std::ifstream& file, const std::string& fname){
            //I read the data
            file.seekg (0, std::ios::beg);
            file.read ((char*)chunk->ptr(), sizeof(value_type)*chunk->size());
            //and remove the file
            if( remove(fname.c_str()) != 0 ){
                throw std::runtime_error("error deleting the cached chunk!");
            }
        }

    private:
        value_type  _init_val;
        std::string _directory;
        uint32_t    _id;
        uint64_t    _max_loaded_chunks;
        uint64_t    _tick;
        typedef std::pair<uint64_t,uint64_t> history_value_type;
        std::priority_queue<history_value_type,std::vector<history_value_type>,std::greater<history_value_type>> _history;
        ChunkedMemory<value_type>* _mem;
    };

}


#endif
