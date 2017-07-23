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

template <typename value_type>
void test_view(){
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

    shape_type idx(3);
    for(idx[0] = 0; idx[0] < shape[0]; ++idx[0]){
        for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
            for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                memory[idx] = idx[0]+idx[1]+idx[2];
                REQUIRE((memory[idx] == idx[0]+idx[1]+idx[2]));
            }
        }
    }

    SECTION("Indexing through slicing"){
        std::cout << "\tIndexing through slicing (" << typeid(value_type).name() << ")" << std::endl;

        idx_ind_type idx_ind;
        idx_ind.push_back(idx_type{0});
        idx_ind.push_back(idx_type{0});
        idx_ind.push_back(idx_type{42});
        auto view = memory.getView(idx_ind);
        REQUIRE((memory[idx_type{0,0,42}] == 42));
        REQUIRE(((*view)[idx_type{0,0,0}] == 42));
        (*view)[idx_type{0,0,0}] = 0;
        REQUIRE((memory[idx_type{0,0,42}] == 0));

    }

    SECTION("Slicing a single plane"){
        std::cout << "\tSlicing a single plane (" << typeid(value_type).name() << ")" << std::endl;

        idx_ind_type idx_ind;
        idx_ind.push_back(idx_type{10});
        idx_ind.push_back(idx_type(50));
        idx_ind.push_back(idx_type(50));
        std::iota(idx_ind[1].begin(),idx_ind[1].end(),0);
        std::iota(idx_ind[2].begin(),idx_ind[2].end(),0);

        auto view = memory.getView(idx_ind);

        shape_type idx(3);
        idx[0] = 0;
        for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
            for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                REQUIRE(((*view)[idx] == 10+idx[1]+idx[2]));
            }
        }

    }

    SECTION("Slicing on multiple planes"){
        std::cout << "\tSlicing on multiple planes (" << typeid(value_type).name() << ")" << std::endl;

        idx_ind_type idx_ind;
        idx_ind.push_back(idx_type{0,10,20,30,40});
        idx_ind.push_back(idx_type(50));
        idx_ind.push_back(idx_type(50));
        std::iota(idx_ind[1].begin(),idx_ind[1].end(),0);
        std::iota(idx_ind[2].begin(),idx_ind[2].end(),0);

        auto view = memory.getView(idx_ind);

        shape_type idx(3);
        for(idx[0] = 0; idx[0] < 5; ++idx[0]){
            for(idx[1] = 0; idx[1] < shape[1]; ++idx[1]){
                for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
                    REQUIRE(((*view)[idx] == idx[0]*10+idx[1]+idx[2]));
                }
            }
        }
    }

    SECTION("Views can be extracted from other views"){
        std::cout << "\tViews can be extracted from other views (" << typeid(value_type).name() << ")" << std::endl;

        idx_ind_type idx_ind_0;
        idx_ind_0.push_back(idx_type{10});
        idx_ind_0.push_back(idx_type(50));
        idx_ind_0.push_back(idx_type(50));
        std::iota(idx_ind_0[1].begin(),idx_ind_0[1].end(),0);
        std::iota(idx_ind_0[2].begin(),idx_ind_0[2].end(),0);

        idx_ind_type idx_ind_1(idx_ind_0);
        idx_ind_1[0][0] = 0;
        idx_ind_1[1].resize(1);
        idx_ind_1[1][0] = 20;

        auto view_0 = memory.getView(idx_ind_0);
        auto view_1 = view_0->getView(idx_ind_1);

        shape_type idx(3);
        idx[0] = 0;
        idx[0] = 0;
        for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
            REQUIRE(((*view_1)[idx] == 10+20+idx[2]));
        }

        idx_ind_type idx_ind_2(idx_ind_1);
        idx_ind_2[1][0] = 0;
        idx_ind_2[2].resize(1);
        idx_ind_2[2][0] = 30;
        auto view_2 = view_1->getView(idx_ind_2);
        REQUIRE(((*view_2)[shape_type{0,0,0}] == 10+20+30));

    }

    SECTION("Views can be extracted from other views with sub space selection"){
        std::cout << "\tViews can be extracted from other views with sub space selection (" << typeid(value_type).name() << ")" << std::endl;

        idx_ind_type idx_ind_0;
        idx_ind_0.push_back(idx_type{10});
        idx_ind_0.push_back(idx_type(50));
        idx_ind_0.push_back(idx_type(50));
        std::iota(idx_ind_0[1].begin(),idx_ind_0[1].end(),0);
        std::iota(idx_ind_0[2].begin(),idx_ind_0[2].end(),0);

        idx_ind_type idx_ind_1(idx_ind_0);
        idx_ind_1[0][0] = 0;
        idx_ind_1[1].resize(1);
        idx_ind_1[1][0] = 20;
        idx_ind_1.erase(idx_ind_1.begin());

        auto view_0 = memory.getView(idx_ind_0);
        auto view_1 = view_0->getViewSubSpace(idx_ind_1);

        shape_type idx(3);
        idx[0] = 0;
        idx[0] = 0;
        for(idx[2] = 0; idx[2] < shape[2]; ++idx[2]){
            REQUIRE(((*view_1)[idx] == 10+20+idx[2]));
        }

        idx_ind_type idx_ind_2(idx_ind_1);
        idx_ind_2[1].resize(1);
        idx_ind_2[1][0] = 30;
        idx_ind_2.erase(idx_ind_2.begin());
        auto view_2 = view_1->getViewSubSpace(idx_ind_2);
        REQUIRE(((*view_2)[shape_type{0,0,0}] == 10+20+30));

    }


}

TEST_CASE( "View works as expected on a single node - float", "[Chunk]" ) {
    test_view<float>();
}
TEST_CASE( "View works as expected on a single node - double", "[Chunk]" ) {
    test_view<double>();
}
TEST_CASE( "View works as expected on a single node - uint32_t", "[Chunk]" ) {
    test_view<uint32_t>();
}
TEST_CASE( "View works as expected on a single node - uint64_t", "[Chunk]" ) {
    test_view<uint64_t>();
}
