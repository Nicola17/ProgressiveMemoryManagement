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

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

#include "pmm/chunked_memory.h"
#include "scoped_timers.h"


double cxx_func(int repeat, uint64_t size){
    double time;
    std::cout << "C++ linear chunk" << std::endl;
    {
        utils::ScopedTimer<double> timer(time);
        for(int r = 0; r < repeat; ++r){
            typedef double value_type;
            typedef pmm::ChunkedMemory<value_type> mem_type;
            typedef typename mem_type::shape_type shape_type;
            typedef typename mem_type::idx_type idx_type;
            typedef typename mem_type::chunk_mat_idx_type chunk_mat_idx_type;
            typedef typename mem_type::chunk_linear_idx_type chunk_linear_idx_type;

            std::vector<value_type> vec(size);
            auto vec_ptr = vec.data();
            int iota = 0;
            for(uint64_t idx = 0; idx < size; ++idx){
                vec_ptr[idx] = iota;
                ++iota;
            }
        }
    }
    std::cout << "\ttime: " << time/repeat << std::endl;
    return time/repeat;
}

double chunks(int repeat, uint64_t size){
    double time;
    std::cout << "1000 chunks" << std::endl;
    {
        utils::ScopedTimer<double> timer(time);
        for(int r = 0; r < repeat; ++r){
            typedef double value_type;
            typedef pmm::ChunkedMemory<value_type> mem_type;
            typedef typename mem_type::shape_type shape_type;
            typedef typename mem_type::idx_type idx_type;
            typedef typename mem_type::chunk_mat_idx_type chunk_mat_idx_type;
            typedef typename mem_type::chunk_linear_idx_type chunk_linear_idx_type;

            mem_type memory;
            shape_type shape{size};
            shape_type chunk_shape{1000};
            memory.setShape(shape,chunk_shape);

            int iota = 0;
            shape_type idx(1);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                memory[idx] = iota;
                ++iota;
            }
        }
    }
    std::cout << "\ttime: " << time/repeat << std::endl;
    return (time/repeat);
}
double single_chunk(int repeat, uint64_t size){
    double time;
    std::cout << "Single chunk" << std::endl;
    {
        utils::ScopedTimer<double> timer(time);
        for(int r = 0; r < repeat; ++r){
            typedef double value_type;
            typedef pmm::ChunkedMemory<value_type> mem_type;
            typedef typename mem_type::shape_type shape_type;
            typedef typename mem_type::idx_type idx_type;
            typedef typename mem_type::chunk_mat_idx_type chunk_mat_idx_type;
            typedef typename mem_type::chunk_linear_idx_type chunk_linear_idx_type;

            mem_type memory;
            shape_type shape{size};
            shape_type chunk_shape{size};
            memory.setShape(shape,chunk_shape);

            int iota = 0;
            shape_type idx(1);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                memory[idx] = iota;
                ++iota;
            }
        }
    }
    std::cout << "\ttime: " << time/repeat << std::endl;
    return (time/repeat);
}

int main(int argc, char *argv[])
{
	try{

        int repeat = 100;
        uint64_t size = 1000000;
        double cxxt = cxx_func(repeat, size);
        double cht = chunks(repeat, size);
        double scht = single_chunk(repeat, size);
        std::cout << "1x - " << cht/cxxt << "x - " << scht/cxxt << "x" << std::endl;

    }
    catch(std::logic_error& ex){ std::cout << "Logic error: " << ex.what() << std::endl;}
    catch(std::runtime_error& ex){ std::cout << "Runtime error: " << ex.what() << std::endl;}
    catch(...){ std::cout << "An unknown error occurred" << std::endl;;}
}
