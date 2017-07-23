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

#ifndef MINI_BATCH_KMEANS_H
#define MINI_BATCH_KMEANS_H

#include <stdint.h>
#include <cassert>
#include <vector>
#include "math_utils.h"
#include "sgd_kmeans.h"


template <typename scalar_type>
class MiniBatchKMeans{
public:
    MiniBatchKMeans(unsigned int dimensionality, unsigned int num_clusters, scalar_type* centroids, unsigned int eta = 1):
        _dimensionality(dimensionality),
        _num_clusters(num_clusters),
        _centroids(centroids),
        _eta(eta),
        _initialized_cluster(0)
    {
        _center_counter.resize(num_clusters,0);
    }

    void doAnIteration(const std::vector<scalar_type*>& data);
    const std::vector<uint64_t>& center_counter()const{return _center_counter;}

private:
    scalar_type* _centroids;
    unsigned int _dimensionality;
    unsigned int _num_clusters;
    unsigned int _eta;
    unsigned int _initialized_cluster;
    std::vector<uint64_t> _center_counter;
    std::vector<uint64_t> _center_cache;
};

template <typename scalar_type>
void MiniBatchKMeans<scalar_type>::doAnIteration(const std::vector<scalar_type*>& data){
    auto batch_size = data.size();
    if(_center_cache.size() != batch_size){
        _center_cache.resize(batch_size);
    }

    //the first iterations are used for the initialization of the centroids
    if(_initialized_cluster < _num_clusters){
        for(int b = 0; b < batch_size && _initialized_cluster < _num_clusters; ++b){
            for(unsigned int d = 0; d < _dimensionality; ++d){
                _centroids[_initialized_cluster*_dimensionality+d] = data[b][d];
            }
            ++_initialized_cluster;
        }
        return;
    }

    //Caching the nearest cluster
    for(int b = 0; b < batch_size; ++b){
        double min_distance(std::numeric_limits<double>::max());
        uint64_t nearest_cluster = 0;
        //computing the nearest cluster
        for(unsigned int c = 0; c < _num_clusters; ++c){
            auto distance = utils::euclideanDistanceSquared(data[b],data[b]+_dimensionality,_centroids+c*_dimensionality,_centroids+(c+1)*_dimensionality);
            if(min_distance > distance){
                min_distance = distance;
                nearest_cluster = c;
            }
        }
        _center_cache[b] = nearest_cluster;
    }

    //Update centroid positions
    for(int b = 0; b < batch_size; ++b){
        uint64_t nearest_cluster = _center_cache[b];
        ++_center_counter[nearest_cluster];
        double learning_rate = 1./_center_counter[nearest_cluster]/_eta;
        //c = (1-learn)*c + learn*x
        for(unsigned int d = 0; d < _dimensionality; ++d){
            _centroids[nearest_cluster*_dimensionality+d] = (1-learning_rate)*_centroids[nearest_cluster*_dimensionality+d] + learning_rate*data[b][d];
        }
    }
}


#endif
