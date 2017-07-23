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
#include "pmm/chunk.h"
#include <stdint.h>
#include <iostream>

template <typename value_type>
void test_chunk(){

    SECTION("Proper initialization"){
        std::cout << "\tProper initialization (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk;
        REQUIRE(chunk.ptr() == nullptr);
        REQUIRE(chunk.size() == 0);
        chunk.resize(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
    }

    SECTION("Constructor initialization"){
        std::cout << "\tConstructor initialization (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
    }

    SECTION("Multiple resizes (1)"){
        std::cout << "\tMultiple resizes (1) (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk;
        REQUIRE(chunk.ptr() == nullptr);
        REQUIRE(chunk.size() == 0);
        chunk.resize(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
        chunk.resize(200);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 200);
    }

    SECTION("Multiple resizes (2)"){
        std::cout << "\tMultiple resizes (2) (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk;
        REQUIRE(chunk.ptr() == nullptr);
        REQUIRE(chunk.size() == 0);
        chunk.resize(200);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 200);
        chunk.resize(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
    }

    SECTION("Resize to 0"){
        std::cout << "\tResize to 0 (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
        chunk.resize(0);
        REQUIRE(chunk.ptr() == nullptr);
        REQUIRE(chunk.size() == 0);
    }

    SECTION("Contained values are preserved after a resize (1)"){
        std::cout << "\tContained values are preserved after a resize (1) (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
        chunk.ptr()[0] = 1;
        chunk.ptr()[20] = 2;
        REQUIRE(chunk.ptr()[0] == 1);
        REQUIRE(chunk.ptr()[20] == 2);
        chunk.resize(30);
        REQUIRE(chunk.ptr()[0] == 1);
        REQUIRE(chunk.ptr()[20] == 2);
    }

    SECTION("Contained values are preserved after a resize (2)"){
        std::cout << "\tContained values are preserved after a resize (2) (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
        chunk.ptr()[0] = 1;
        chunk.ptr()[20] = 2;
        REQUIRE(chunk.ptr()[0] == 1);
        REQUIRE(chunk.ptr()[20] == 2);
        chunk.resize(200);
        REQUIRE(chunk.ptr()[0] == 1);
        REQUIRE(chunk.ptr()[20] == 2);
    }

    SECTION("Operator[] works properly"){
        std::cout << "\tOperator[] works properly (" << typeid(value_type).name() << ")" << std::endl;
        pmm::Chunk<value_type> chunk(100);
        REQUIRE(chunk.ptr() != nullptr);
        REQUIRE(chunk.size() == 100);
        chunk.ptr()[0] = 1;
        chunk.ptr()[20] = 2;
        REQUIRE(chunk[0] == 1);
        REQUIRE(chunk[20] == 2);
        chunk[0] = 2;
        chunk[20] = 1;
        REQUIRE(chunk[0] == 2);
        REQUIRE(chunk[20] == 1);
    }


}

TEST_CASE( "Chunk works as expected - float", "[Chunk]" ) {
    test_chunk<float>();
}
TEST_CASE( "Chunk works as expected - double", "[Chunk]" ) {
    test_chunk<double>();
}
TEST_CASE( "Chunk works as expected - uint32_t", "[Chunk]" ) {
    test_chunk<uint32_t>();
}
TEST_CASE( "Chunk works as expected - uint64_t", "[Chunk]" ) {
    test_chunk<uint64_t>();
}

