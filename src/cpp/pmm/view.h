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

#ifndef VIEW_H
#define VIEW_H

#include "pmm/chunked_memory.h"
#include <vector>
#include <memory>
#include <iostream>

namespace pmm{

    template <typename value_type>
    class ChunkedMemory;

    //! Chunked memory
    /*!
      \author Nicola Pezzotti
    */
    template <typename value_type>
    class View{
    public:
        typedef uint64_t uint64_type;
        typedef std::vector<uint64_t> uint64_vec_type;
        typedef std::vector<uint64_vec_type> idx_ind_type;

    public:
        //!Constructs the view with the provided indirection
        View(){}
        View(const idx_ind_type& idx_ind):_idx_ind(idx_ind){}
        //!Access the memory
        //TODO CRITICAL for performance to optimize
        value_type& operator[](const uint64_vec_type& idx){
            assert(idx.size() == _idx_ind.size());
            uint64_vec_type new_idx(idx);
            for(int i = 0; i < new_idx.size(); ++i){
                assert(idx[i] < _idx_ind[i].size());
                new_idx[i] = _idx_ind[i][idx[i]];
            }
            return (*_memory_ptr)[new_idx];
        }

        //TODO CRITICAL for performance to optimize
        value_type& dataAtSubSpace(const uint64_vec_type& idx){ //NAME?!?
            uint64_vec_type new_idx(_idx_ind.size());
            int k = 0;
            for(int i = 0; i < new_idx.size(); ++i, ++k){
                if(_idx_ind[i].size() == 1){
                    new_idx[i] = _idx_ind[i][0];
                    --k;
                    continue;
                }
                assert(idx[k] < _idx_ind[i].size());
                new_idx[i] = _idx_ind[i][idx[k]];
            }
            return (*_memory_ptr)[new_idx];
        }

        //TODO CRITICAL for performance to optimize
        //TODO WEIRD FOR CYTHON
        void setDataAtSubSpace(const uint64_vec_type& idx, value_type y){ //NAME?!?
            uint64_vec_type new_idx(_idx_ind.size());
            int k = 0;
            for(int i = 0; i < new_idx.size(); ++i, ++k){
                if(_idx_ind[i].size() == 1){
                    new_idx[i] = _idx_ind[i][0];
                    --k;
                    continue;
                }
                assert(idx[k] < _idx_ind[i].size());
                new_idx[i] = _idx_ind[i][idx[k]];
            }
            (*_memory_ptr)[new_idx] = y;
        }

        std::shared_ptr<View<value_type>> getView(const idx_ind_type& idx_ind){
            idx_ind_type new_idx_ind(idx_ind);
            assert(idx_ind.size() == _idx_ind.size());
            for(int j = 0; j < new_idx_ind.size(); ++j){
                for(int i = 0; i < new_idx_ind[j].size(); ++i){
                    assert(idx_ind[j][i] < _idx_ind[j].size());
                    new_idx_ind[j][i] = _idx_ind[j][idx_ind[j][i]];
                }
            }
            auto ptr = std::make_shared<View<value_type>>(new_idx_ind);
            ptr->_memory_ptr = _memory_ptr;
            return ptr;
        }
        std::shared_ptr<View<value_type>> getViewSubSpace(const idx_ind_type& idx_ind){
            idx_ind_type new_idx_ind(idx_ind);
            for(int i = 0; i < _idx_ind.size(); ++i){
                if(_idx_ind[i].size() == 1){
                    new_idx_ind.insert(new_idx_ind.begin()+i,std::vector<uint64_t>(1,0));
                }
            }
            return getView(new_idx_ind);
        }

        const idx_ind_type& idx_ind(){return _idx_ind;}
        uint64_t size(uint64_t i){return 0;}//TODO

    private:
        idx_ind_type _idx_ind;

    public://TEMP
        ChunkedMemory<value_type>* _memory_ptr; //shared ptr?
    };

}

#endif
