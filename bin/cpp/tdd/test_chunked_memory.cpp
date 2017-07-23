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
#include <stdint.h>
#include <iostream>

template <typename value_type>
void test_chunked_memory(){
    typedef pmm::ChunkedMemory<value_type> mem_type;
    typedef typename mem_type::shape_type shape_type;
    typedef typename mem_type::idx_type idx_type;
    typedef typename mem_type::chunk_mat_idx_type chunk_mat_idx_type;
    typedef typename mem_type::chunk_linear_idx_type chunk_linear_idx_type;

    SECTION("Shape is saved in construction"){
        std::cout << "\tShape is saved in construction (" << typeid(value_type).name() << ")" << std::endl;
        shape_type shape{100,100,100};
        shape_type chunk_shape{10,10,10};
        shape_type chunk_grid_shape{10,10,10};
        mem_type memory(shape,chunk_shape);
        REQUIRE(memory.shape() == shape);
        REQUIRE(memory.chunk_shape() == chunk_shape);
        REQUIRE(memory.chunk_grid_shape() == chunk_grid_shape);
        REQUIRE(memory.chunk_byte_size() == (10*10*10*sizeof(value_type)));
    }

    SECTION("Shape is saved"){
        std::cout << "\tShape is saved (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{100,100,100};
        shape_type chunk_shape{10,10,10};
        shape_type chunk_grid_shape{10,10,10};
        memory.setShape(shape,chunk_shape);
        REQUIRE(memory.shape() == shape);
        REQUIRE(memory.chunk_shape() == chunk_shape);
        REQUIRE(memory.chunk_grid_shape() == chunk_grid_shape);
        REQUIRE(memory.chunk_byte_size() == (10*10*10*sizeof(value_type)));
    }

    SECTION("Chunk shape must be bigger than 0"){
        std::cout << "\tChunk shape must be bigger than 0 (" << typeid(value_type).name() << ")" << std::endl;
        shape_type shape{100,100,100};
        shape_type chunk_shape{0,10,10};
        mem_type memory;
        REQUIRE_THROWS(memory.setShape(shape,chunk_shape));
    }

    SECTION("Chunk grid may be bigger than the input"){
        std::cout << "\tChunk grid may be bigger than the input (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{101,101,101};
        shape_type chunk_shape{10,10,10};
        shape_type chunk_grid_shape{11,11,11};
        memory.setShape(shape,chunk_shape);
        REQUIRE(memory.shape() == shape);
        REQUIRE(memory.chunk_shape() == chunk_shape);
        REQUIRE(memory.chunk_grid_shape() == chunk_grid_shape);
    }

    SECTION("Chunk grid may be empty"){
        std::cout << "\tChunk grid may be empty (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{0,0,0};
        shape_type chunk_shape{10,10,10};
        shape_type chunk_grid_shape{0,0,0};
        memory.setShape(shape,chunk_shape);
        REQUIRE(memory.shape() == shape);
        REQUIRE(memory.chunk_shape() == chunk_shape);
        REQUIRE(memory.chunk_grid_shape() == chunk_grid_shape);

        memory.reshape(shape_type{20,20,30});
        REQUIRE((memory.chunk_grid_shape() == shape_type{2,2,3}));
    }

    SECTION("Basic indices conversion (1)"){
        std::cout << "\tBasic indices conversion (1) (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{100,100,100};
        shape_type chunk_shape{10,10,10};
        memory.setShape(shape,chunk_shape);
        //Index to chunk matricial index
        REQUIRE((memory.idxToMatChunkIdx(idx_type{0,0,0})    == chunk_mat_idx_type{0,0,0}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{1,1,1})    == chunk_mat_idx_type{0,0,0}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{9,9,9})    == chunk_mat_idx_type{0,0,0}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{10,9,9})   == chunk_mat_idx_type{1,0,0}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{10,10,10}) == chunk_mat_idx_type{1,1,1}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{99,99,99}) == chunk_mat_idx_type{9,9,9}));

        //Chunk matricial index to chunk linear index
        REQUIRE((memory.matToLinearChunkIdx(idx_type{0,0,0}) == 0));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{9,9,9}) == 999));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{0,0,1}) == 1));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{0,1,0}) == 10));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{1,0,0}) == 100));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{1,1,1}) == 111));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{9,1,1}) == 911));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{9,0,9}) == 909));

        //Chunk linear index to chunk matricial index
        REQUIRE((memory.linearToMatChunkIdx(  0) == idx_type{0,0,0}));
        REQUIRE((memory.linearToMatChunkIdx(999) == idx_type{9,9,9}));
        REQUIRE((memory.linearToMatChunkIdx(  1) == idx_type{0,0,1}));
        REQUIRE((memory.linearToMatChunkIdx( 10) == idx_type{0,1,0}));
        REQUIRE((memory.linearToMatChunkIdx(100) == idx_type{1,0,0}));
        REQUIRE((memory.linearToMatChunkIdx(111) == idx_type{1,1,1}));
        REQUIRE((memory.linearToMatChunkIdx(911) == idx_type{9,1,1}));
        REQUIRE((memory.linearToMatChunkIdx(909) == idx_type{9,0,9}));

        //Index to in-chunk index
        REQUIRE(memory.idxInChunk(idx_type{0,0,0},memory.idxToMatChunkIdx(idx_type{0,0,0}))    == 0);
        REQUIRE(memory.idxInChunk(idx_type{1,1,1},memory.idxToMatChunkIdx(idx_type{1,1,1}))    == 111);
        REQUIRE(memory.idxInChunk(idx_type{9,9,9},memory.idxToMatChunkIdx(idx_type{9,9,9}))    == 999);
        REQUIRE(memory.idxInChunk(idx_type{10,9,9},memory.idxToMatChunkIdx(idx_type{10,9,9}))   == 99);
        REQUIRE(memory.idxInChunk(idx_type{10,10,10},memory.idxToMatChunkIdx(idx_type{10,10,10})) == 0);
        REQUIRE(memory.idxInChunk(idx_type{99,99,99},memory.idxToMatChunkIdx(idx_type{99,99,99})) == 999);


    }

    SECTION("Basic indices conversion (2)"){
        std::cout << "\tBasic indices conversion (2) (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{100,100,100};
        shape_type chunk_shape{10,1,5};
        memory.setShape(shape,chunk_shape);
        //Index to chunk matricial index
        REQUIRE((memory.idxToMatChunkIdx(idx_type{0,0,0})    == chunk_mat_idx_type{0,0,0}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{1,1,1})    == chunk_mat_idx_type{0,1,0}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{9,9,9})    == chunk_mat_idx_type{0,9,1}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{10,9,9})   == chunk_mat_idx_type{1,9,1}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{10,10,10}) == chunk_mat_idx_type{1,10,2}));
        REQUIRE((memory.idxToMatChunkIdx(idx_type{99,99,99}) == chunk_mat_idx_type{9,99,19}));

        //Chunk matricial index to chunk linear index
        REQUIRE((memory.matToLinearChunkIdx(idx_type{0,0,0})   == 0));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{9,99,19}) == 19999));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{0,0,1})   == 1));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{0,1,0})   == 20));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{1,0,0})   == 2000));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{1,1,1})   == 2021));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{9,1,1})   == 18021));
        REQUIRE((memory.matToLinearChunkIdx(idx_type{9,0,9})   == 18009));

        //Chunk linear index to chunk matricial index
        REQUIRE((memory.linearToMatChunkIdx(  0)   == idx_type{0,0,0}));
        REQUIRE((memory.linearToMatChunkIdx(19999) == idx_type{9,99,19}));
        REQUIRE((memory.linearToMatChunkIdx(  1)   == idx_type{0,0,1}));
        REQUIRE((memory.linearToMatChunkIdx( 20)   == idx_type{0,1,0}));
        REQUIRE((memory.linearToMatChunkIdx(2000)  == idx_type{1,0,0}));
        REQUIRE((memory.linearToMatChunkIdx(2021)  == idx_type{1,1,1}));
        REQUIRE((memory.linearToMatChunkIdx(18021) == idx_type{9,1,1}));
        REQUIRE((memory.linearToMatChunkIdx(18009) == idx_type{9,0,9}));
    }

    SECTION("Initially data is uninitialized"){
        std::cout << "\tInitially data is uninitialized (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{100,100,100};
        shape_type chunk_shape{10,10,10};
        memory.setShape(shape,chunk_shape);

        REQUIRE(memory.initialized(idx_type{0,0,0}) == false);
        REQUIRE(memory.initialized(idx_type{10,0,0}) == false);
        REQUIRE(memory.initialized(idx_type{10,10,10}) == false);
        REQUIRE(memory.initialized(idx_type{5,5,5}) == false);

        REQUIRE(memory.inMemory(idx_type{0,0,0}) == false);
        REQUIRE(memory.inMemory(idx_type{10,0,0}) == false);
        REQUIRE(memory.inMemory(idx_type{10,10,10}) == false);
        REQUIRE(memory.inMemory(idx_type{5,5,5}) == false);
    }

    SECTION("Loaded chunks are initialized and in memory"){
        std::cout << "\tLoaded chunks are initialized and in memory (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{100,100,100};
        shape_type chunk_shape{10,10,10};
        memory.setShape(shape,chunk_shape);
        memory.loadInMemory(idx_type{0,0,0});
        memory.loadInMemory(idx_type{10,0,0});
        memory.loadInMemory(idx_type{10,10,10});
        memory.loadInMemory(idx_type{50,50,50});

        REQUIRE(memory.initialized(idx_type{0,0,0}) == true);
        REQUIRE(memory.initialized(idx_type{10,0,0}) == true);
        REQUIRE(memory.initialized(idx_type{10,10,10}) == true);
        REQUIRE(memory.initialized(idx_type{50,50,50}) == true);

        REQUIRE(memory.inMemory(idx_type{0,0,0}) == true);
        REQUIRE(memory.inMemory(idx_type{10,0,0}) == true);
        REQUIRE(memory.inMemory(idx_type{10,10,10}) == true);
        REQUIRE(memory.inMemory(idx_type{50,50,50}) == true);
    }

    SECTION("Iota 3D"){
        std::cout << "\tIota 3D (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{30,30,30};
        shape_type chunk_shape{10,10,10};
        memory.setShape(shape,chunk_shape);

        int iota = 0;
        shape_type idx(3);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    memory[idx] = iota;
                    REQUIRE(memory[idx] == iota);
                    ++iota;
                }
            }
        }

        iota = 0;
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    REQUIRE(memory[idx] == iota);
                    ++iota;
                }
            }
        }
    }

    SECTION("Iota 3D single chunk"){
        std::cout << "\tIota 3D single chunk (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{30,30,30};
        shape_type chunk_shape{30,30,30};
        memory.setShape(shape,chunk_shape);

        int iota = 0;
        shape_type idx(3);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    memory[idx] = iota;
                    REQUIRE(memory[idx] == iota);
                    ++iota;
                }
            }
        }

        iota = 0;
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    REQUIRE(memory[idx] == iota);
                    ++iota;
                }
            }
        }
    }

    SECTION("Iota 3D linear chunk"){
        std::cout << "\tIota 3D linear chunk(" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{30,30,30};
        shape_type chunk_shape{1,1,30};
        memory.setShape(shape,chunk_shape);

        int iota = 0;
        shape_type idx(3);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    memory[idx] = iota;
                    REQUIRE(memory[idx] == iota);
                    ++iota;
                }
            }
        }

        iota = 0;
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    REQUIRE(memory[idx] == iota);
                    ++iota;
                }
            }
        }
    }

    SECTION("C++ linear chunk"){
        std::cout << "\tC++ linear chunk(" << typeid(value_type).name() << ")" << std::endl;
        shape_type shape{30,30,30};

        std::vector<value_type> vec(shape[0]*shape[1]*shape[2]);
        int iota = 0;
        shape_type idx(3);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    uint64_t lin_idx = idx[0]*shape[1]*shape[2] + idx[1]*shape[2] + idx[2];
                    vec[lin_idx] = iota;
                    REQUIRE(vec[lin_idx] == iota);
                    ++iota;
                }
            }
        }

        iota = 0;
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    uint64_t lin_idx = idx[0]*shape[1]*shape[2] + idx[1]*shape[2] + idx[2];
                    REQUIRE(vec[lin_idx] == iota);
                    ++iota;
                }
            }
        }
    }

    SECTION("Multithreaded computation"){
        std::cout << "\tMultithreaded computation(" << typeid(value_type).name() << ")" << std::endl;
        shape_type shape{30,30,30};

        std::vector<value_type> vec(shape[0]*shape[1]*shape[2]);

     #pragma omp parallel for
        for(int o = 0; o < shape[0]; ++o){
            shape_type idx(3);
            idx[0] = o;
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    uint64_t lin_idx = idx[0]*shape[1]*shape[2] + idx[1]*shape[2] + idx[2];
                    #pragma omp critical
                    {
                        vec[lin_idx] = idx[0];
                    }
                }
            }
        }

        shape_type idx(3);
        for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    uint64_t lin_idx = idx[0]*shape[1]*shape[2] + idx[1]*shape[2] + idx[2];
                    REQUIRE(vec[lin_idx] == idx[0]);
                }
            }
        }
    }


    SECTION("Incremental reshape works as expected"){
        std::cout << "\tIncremental reshape works as expected (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{15,20,25};
        shape_type chunk_shape{10,10,10};
        memory.setShape(shape,chunk_shape);
        memory.loadInMemory(idx_type{0,0,0});
        memory.loadInMemory(idx_type{10,0,0});
        memory.loadInMemory(idx_type{10,10,10});

        REQUIRE(memory.initialized(idx_type{0,0,0}) == true);
        REQUIRE(memory.initialized(idx_type{10,0,0}) == true);
        REQUIRE(memory.initialized(idx_type{10,10,10}) == true);

        REQUIRE(memory.inMemory(idx_type{0,0,0}) == true);
        REQUIRE(memory.inMemory(idx_type{10,0,0}) == true);
        REQUIRE(memory.inMemory(idx_type{10,10,10}) == true);

        {
            int iota = 0;
            shape_type idx(3);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                    for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                        memory[idx] = iota;
                        REQUIRE(memory[idx] == iota);
                        ++iota;
                    }
                }
            }

            iota = 0;
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                    for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                        REQUIRE(memory[idx] == iota);
                        ++iota;
                    }
                }
            }
        }

        {//Reuses the alredy allocated chunks
            memory.reshape(shape_type{20,20,30});
            int iota = 0;
            shape_type idx(3);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                    for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                        memory[idx] = iota;
                        REQUIRE(memory[idx] == iota);
                        ++iota;
                    }
                }
            }
            iota = 0;
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                    for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                        REQUIRE(memory[idx] == iota);
                        ++iota;
                    }
                }
            }
        }

        {
            shape_type new_shape{40,40,40};
            memory.reshape(new_shape);
            memory[shape_type{39,39,39}] = 0;

            //The original data should be there and rightly initialized
            int iota = 0;
            shape_type idx(3);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                    for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                        REQUIRE(memory[idx] == iota);
                        ++iota;
                    }
                }
            }

            //Initialization of new data
            iota = 0;
            for(idx[0] = 0; idx[0] < new_shape[0]; ++idx[0]){
                for(idx[1] = 0; idx[1] < new_shape[1]; ++idx[1]){
                    for(idx[2] = 0; idx[2] < new_shape[2]; ++idx[2]){
                        memory[idx] = iota;
                        REQUIRE(memory[idx] == iota);
                        ++iota;
                    }
                }
            }
            //Final check
            iota = 0;
            for(idx[0] = 0; idx[0] < new_shape[0]; ++idx[0]){
                for(idx[1] = 0; idx[1] < new_shape[1]; ++idx[1]){
                    for(idx[2] = 0; idx[2] < new_shape[2]; ++idx[2]){
                        REQUIRE(memory[idx] == iota);
                        ++iota;
                    }
                }
            }
        }

    }

    SECTION("Incremental reshape for linear array works as expected"){
        std::cout << "\tIncremental reshape for linear array  works as expected (" << typeid(value_type).name() << ")" << std::endl;
        mem_type memory;
        shape_type shape{1000};
        shape_type chunk_shape{20};
        memory.setShape(shape,chunk_shape);
        memory.loadInMemory(idx_type{0});

        REQUIRE(memory.initialized(idx_type{0}) == true);
        REQUIRE(memory.inMemory(idx_type{0}) == true);

        {
            int iota = 0;
            shape_type idx(1);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                memory[idx] = iota;
                REQUIRE(memory[idx] == iota);
                ++iota;
            }

            iota = 0;
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                REQUIRE(memory[idx] == iota);
                ++iota;
            }
        }

        {//Reuses the alredy allocated chunks
            memory.reshape(shape_type{1010});
            int iota = 0;
            shape_type idx(1);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                memory[idx] = iota;
                REQUIRE(memory[idx] == iota);
                ++iota;
            }
            iota = 0;
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                REQUIRE(memory[idx] == iota);
                ++iota;
            }
        }

        {
            shape_type new_shape{5000};
            memory.reshape(new_shape);
            memory[shape_type{42}] = 0;
            memory[shape_type{42}] = 42;

            //The original data should be there and rightly initialized
            int iota = 0;
            shape_type idx(1);
            for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
                REQUIRE(memory[idx] == iota);
                ++iota;
            }

            //Initialization of new data
            iota = 0;
            for(idx[0] = 0; idx[0] < new_shape[0]; ++idx[0]){
                memory[idx] = iota;
                REQUIRE(memory[idx] == iota);
                ++iota;
            }
            //Final check
            iota = 0;
            for(idx[0] = 0; idx[0] < new_shape[0]; ++idx[0]){
                REQUIRE(memory[idx] == iota);
                ++iota;
            }
        }

    }

}

TEST_CASE( "Chunked memory works as expected on a single node - float", "[Chunk]" ) {
    test_chunked_memory<float>();
}
TEST_CASE( "Chunked memory works as expected on a single node - double", "[Chunk]" ) {
    test_chunked_memory<double>();
}
TEST_CASE( "Chunked memory works as expected on a single node - uint32_t", "[Chunk]" ) {
    test_chunked_memory<uint32_t>();
}
TEST_CASE( "Chunked memory works as expected on a single node - uint64_t", "[Chunk]" ) {
    test_chunked_memory<uint64_t>();
}
