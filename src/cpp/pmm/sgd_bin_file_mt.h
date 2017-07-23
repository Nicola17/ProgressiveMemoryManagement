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

#ifndef SGD_BIN_FILE_MT_H
#define SGD_BIN_FILE_MT_H

#include <stdint.h>
#include "pmm/chunked_memory.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <deque>
#include <random>
#include <thread>
#include <mutex>

namespace pmm{

    //! Chunked memory
    /*!
        \author Nicola Pezzotti
    */
    template <typename value_type>
    class SGDBinFileMT{
    public:
        typedef typename ChunkedMemory<value_type>::shape_type shape_type;
        typedef typename ChunkedMemory<value_type>::chunk_type chunk_type;

    public:
        SGDBinFileMT(const std::string& str, const shape_type& shape);
        value_type* getRandomData();

        void getMultipleDataPoints(std::vector<value_type>& ptrs, unsigned int n); // this is quite dangerous... TODO

    private:
        uint64_t getRandomIdx();
        void loaderFunc();

    private:
        shape_type _shape;
        shape_type _chunk_shape;
        ChunkedMemory<value_type>   _memory;
        std::deque<chunk_type*>     _loaded_chunks;
        std::deque<uint64_t>        _loaded_chunks_idx;

        //Random number generator
        std::default_random_engine _generator;
        std::uniform_int_distribution<> _distribution_pnts;
        std::uniform_int_distribution<> _distribution_chunks;
        std::uniform_int_distribution<> _distribution_in_chunks;

        //parameters
        unsigned int _max_chunks_in_memory;
        unsigned int _samples_per_chunk;
        unsigned int _n_sampled;

        //File
        std::ifstream _in_file;

        std::thread _loader;
        std::mutex  _mem_mutex;
        std::mutex  _queue_mutex;

    };

}

#endif
