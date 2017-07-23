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

#include "catch.hpp"
#include "pmm/chunked_memory.h"
#include "pmm/view.h"
#include <stdint.h>
#include <iostream>
#include "pmm/policy_rw_growing.h"
#include "pmm/policy_rw_chached.h"
#include "pmm/policy_r_bin_file.h"
#include <fstream>

template <typename value_type>
void test_growing(){
    typedef pmm::ChunkedMemory<value_type> mem_type;
    typedef typename mem_type::shape_type shape_type;
    typedef typename mem_type::idx_type idx_type;
    typedef typename mem_type::chunk_mat_idx_type chunk_mat_idx_type;
    typedef typename mem_type::chunk_linear_idx_type chunk_linear_idx_type;
    typedef typename mem_type::view_type view_type;
    typedef typename view_type::idx_ind_type idx_ind_type;

    shape_type shape{50,50,50};
    shape_type chunk_shape{10,10,10};
    mem_type memory(shape,chunk_shape);


    SECTION("Test RW Growing (0)"){
        std::cout << "\tRW Growing policy (" << typeid(value_type).name() << ")" << std::endl;
        value_type def_val = 42;
        pmm::ReadWriteGrowingPolicy<value_type> policy(def_val);
        memory.set_memory_policy(&policy);
        shape_type idx(3);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    value_type v = memory[idx];
                    REQUIRE(v == def_val);
                }
            }
        }
    }
}

template <typename value_type>
void test_cached(){
    typedef pmm::ChunkedMemory<value_type> mem_type;
    typedef typename mem_type::shape_type shape_type;
    typedef typename mem_type::idx_type idx_type;
    typedef typename mem_type::chunk_mat_idx_type chunk_mat_idx_type;
    typedef typename mem_type::chunk_linear_idx_type chunk_linear_idx_type;
    typedef typename mem_type::view_type view_type;
    typedef typename view_type::idx_ind_type idx_ind_type;

    shape_type shape{50,50,50};
    shape_type chunk_shape{10,10,10};
    mem_type memory(shape,chunk_shape);


    SECTION("Test RW Cached (0)"){
        std::cout << "\tRW Cached policy (" << typeid(value_type).name() << ")" << std::endl;
        value_type def_val = 42;
        uint64_t max_chunks = 5;
        pmm::ReadWriteCachedPolicy<value_type> policy("",max_chunks,def_val);
        memory.set_memory_policy(&policy);
        shape_type idx(3);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    value_type v = memory[idx];
                    REQUIRE(v == def_val);
                    v = idx[0]+idx[1]+idx[2];
                    memory[idx] = v;
                    REQUIRE(v == memory[idx]);
                    REQUIRE(memory.loadedChunks() <= max_chunks);
                }
            }
        }
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    const value_type v = idx[0]+idx[1]+idx[2];
                    REQUIRE(v == memory[idx]);
                    REQUIRE(memory.loadedChunks() <= max_chunks);
                }
            }
        }
    }
}

template <typename value_type>
void test_read_file(){
    typedef pmm::ChunkedMemory<value_type> mem_type;
    typedef typename mem_type::shape_type shape_type;
    typedef typename mem_type::idx_type idx_type;
    typedef typename mem_type::chunk_mat_idx_type chunk_mat_idx_type;
    typedef typename mem_type::chunk_linear_idx_type chunk_linear_idx_type;
    typedef typename mem_type::view_type view_type;
    typedef typename view_type::idx_ind_type idx_ind_type;

    shape_type shape{10000,10};
    shape_type chunk_shape{100,10};
    mem_type memory(shape,chunk_shape);


    SECTION("Test R Binary File (0)"){
        std::cout << "\tR Binary file policy (" << typeid(value_type).name() << ")" << std::endl;
        std::string filename("temp.bin");
        std::ofstream f(filename.c_str(),std::ios::binary);
        REQUIRE_FALSE(f.fail());

        shape_type idx(2);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                value_type v = idx[0];
                f.write((char*)&v,sizeof(value_type));
            }
        }

        uint64_t max_chunks = 5;
        pmm::ReadFromFileBinaryPolicy<value_type> policy(filename,max_chunks);
        memory.set_memory_policy(&policy);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                const value_type v = idx[0];
                REQUIRE(v == memory[idx]);
                REQUIRE(memory.loadedChunks() <= max_chunks);
            }
        }
    }
}

TEST_CASE( "Policy tests - float", "[Chunk]" ) {
    test_growing<float>();
    test_cached<float>();
    test_read_file<float>();
}
TEST_CASE( "Policy tests - double", "[Chunk]" ) {
    test_growing<double>();
    test_cached<double>();
    test_read_file<double>();
}
TEST_CASE( "Policy tests - uint32_t", "[Chunk]" ) {
    test_growing<uint32_t>();
    test_cached<uint32_t>();
    test_read_file<uint32_t>();
}
TEST_CASE( "Policy tests - uint64_t", "[Chunk]" ) {
    test_growing<uint64_t>();
    test_cached<uint64_t>();
    test_read_file<uint64_t>();
}
