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

#ifndef CHUNK_INL
#define CHUNK_INL

#include "pmm/chunk.h"
#include <cstdlib>
#include <cstring>

namespace pmm{

    template <typename value_type>
    Chunk<value_type>::Chunk():_size(0),_ptr(nullptr){

    }

    template <typename value_type>
    Chunk<value_type>::Chunk(uint64_t size):_size(size){
        if(_size == 0){
            return;
        }
        int n_bytes = sizeof(value_type) * _size;
        _ptr = (value_type*) std::malloc(n_bytes);
        //std::memset(_ptr,0,n_bytes); //now is implemented in the policies
    }

    template <typename value_type>
    Chunk<value_type>::~Chunk(){
        std::free(_ptr);
    }

    template <typename value_type>
    void Chunk<value_type>::Chunk::resize(uint64_t size){
        if(size == 0){
            _size = size;
            std::free(_ptr);
            _ptr = nullptr;
            return;
        }
        value_type* old_ptr  = _ptr;
        uint64_t    old_size = _size;
        _size = size;

        _ptr = (value_type*) std::malloc(sizeof(value_type) * _size);
        std::memset(_ptr,0,sizeof(value_type) * _size); //not sure about this

        std::memcpy(_ptr,old_ptr,sizeof(value_type) * (_size<old_size?_size:old_size));

        std::free(old_ptr);
    }


}

#endif
