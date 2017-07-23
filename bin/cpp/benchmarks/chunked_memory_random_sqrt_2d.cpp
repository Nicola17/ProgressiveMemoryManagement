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


void cxx_func(int repeat, int num){
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
            shape_type shape{1000,1000};

            std::vector<value_type> vec(shape[0]*shape[1]);
            shape_type idx(2);
            for(int i = 0; i < num; ++i){
                for(int s = 0; s < shape.size(); ++s){
                    idx[s] = rand()%shape[s];
                }
                uint64_t lin_idx = idx[0]*shape[1] + idx[1];
                vec[lin_idx] = std::sqrt(std::rand()%1000);
            }
        }
    }
    std::cout << "\ttime: " << time/repeat << std::endl;
}

void chunks_3d(int repeat, int num){
    double time;
    std::cout << "50x50 chunks" << std::endl;
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
            shape_type shape{1000,1000};
            shape_type chunk_shape{50,50};
            memory.setShape(shape,chunk_shape);

            shape_type idx(2);
            for(int i = 0; i < num; ++i){
                for(int s = 0; s < shape.size(); ++s){
                    idx[s] = rand()%shape[s];
                }
                memory[idx] = std::sqrt(std::rand()%1000);
            }
        }
    }
    std::cout << "\ttime: " << time/repeat << std::endl;
}
void single_chunk(int repeat, int num){
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
            shape_type shape{1000,1000};
            shape_type chunk_shape{1000,1000};
            memory.setShape(shape,chunk_shape);

            shape_type idx(2);
            for(int i = 0; i < num; ++i){
                for(int s = 0; s < shape.size(); ++s){
                    idx[s] = rand()%shape[s];
                }
                memory[idx] = std::sqrt(std::rand()%1000);
            }
        }
    }
    std::cout << "\ttime: " << time/repeat << std::endl;
}

void linear_chunks(int repeat, int num){
    double time;
    std::cout << "1x1000" << std::endl;
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
            shape_type shape{1000,1000};
            shape_type chunk_shape{1,1000};
            memory.setShape(shape,chunk_shape);

            shape_type idx(2);
            for(int i = 0; i < num; ++i){
                for(int s = 0; s < shape.size(); ++s){
                    idx[s] = rand()%shape[s];
                }
                memory[idx] = std::sqrt(std::rand()%1000);
            }
        }
    }
    std::cout << "\ttime: " << time/repeat << std::endl;
}

int main(int argc, char *argv[])
{
	try{
        int repeat = 10;
        int num = 1000000;
        cxx_func(repeat, num);
        chunks_3d(repeat, num);
        single_chunk(repeat, num);
        linear_chunks(repeat, num);
    }
    catch(std::logic_error& ex){ std::cout << "Logic error: " << ex.what() << std::endl;}
    catch(std::runtime_error& ex){ std::cout << "Runtime error: " << ex.what() << std::endl;}
    catch(...){ std::cout << "An unknown error occurred" << std::endl;;}
}
