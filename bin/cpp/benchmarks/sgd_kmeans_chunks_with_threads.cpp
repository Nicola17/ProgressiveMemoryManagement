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
#include <random>
#include <iomanip>
#include <cassert>
#include "pmm/sgd_bin_file_mt.h"
#include "scoped_timers.h"
#include "sgd_kmeans.h"

int main(int argc, char *argv[])
{
    try{
        typedef float scalar_type;

        if(argc != 5){
            std::cout << "Wrong number of parameters!" << std::endl;
            std::cout << "1: file name" << std::endl;
            std::cout << "2: # dimensions" << std::endl;
            std::cout << "3: # points" << std::endl;
            std::cout << "4: # clusters" << std::endl;
            return 1;
        }

        const std::string file_name(argv[1]);
        const uint64_t n_dim(std::atoi(argv[2]));
        const uint64_t n_pnt(std::atoi(argv[3]));
        const uint64_t n_cls(std::atoi(argv[4]));
        const uint64_t max_it = 10e5;
        const bool log_errors = true;

        std::default_random_engine generator;
        std::uniform_int_distribution<> distribution_int(0, n_pnt-1);

        std::vector<scalar_type> centroids(n_cls*n_dim);
        SGDKMeans<scalar_type> kmeans(n_dim,n_cls,centroids.data(),max_it*0.33);

        pmm::SGDBinFileMT<scalar_type>::shape_type shape = {n_pnt,n_dim};
        pmm::SGDBinFileMT<scalar_type> memory(file_name,shape);

        int log_iter = 1;
        std::vector<std::pair<uint64_t,double>> log;
        log.reserve(10000);
        double time;
        {
            utils::ScopedTimer<double> timer(time);
            uint64_t it = 0;
            while(++it <= max_it){
                kmeans.doAnIteration(memory.getRandomData());

                if(log_errors){
                    if(it == 10){
                        log_iter = 2;
                    }else if(it == 20){
                        log_iter = 5;
                    }else if(it == 100){
                        log_iter = 50;
                    }else if(it == 1000){
                        log_iter = 5000;
                    }else if(it == 100000){
                        log_iter = 50000;
                    }
                    if((it%log_iter) == 0){
                        double error = computeErrorFromFile(n_dim,n_cls,n_pnt,centroids.data(),argv[1]);
                        std::cout << it << "/" << max_it << ": " << error << std::endl;
                        log.push_back(std::make_pair(it,error));
                    }
                }
            }
        }


        for (int i = 0; i < n_cls; ++i){
            std::cout << kmeans.center_counter()[i] << " ||| ";
            for(int d = 0; d < n_dim; ++d){
                std::cout << std::setprecision(2) << centroids[i*n_dim+d] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << "TIME: " << std::setprecision(9) << time/1000 << std::endl;

        if(log_errors){
            std::ofstream csv_file;
            csv_file.open("sgd_kmeans_chunks.csv");
            for(auto& l: log){
                csv_file << l.first << "," << l.second << std::endl;
            }
        }
    }
    catch(std::logic_error& ex){ std::cout << "Logic error: " << ex.what() << std::endl;}
    catch(std::runtime_error& ex){ std::cout << "Runtime error: " << ex.what() << std::endl;}
    catch(...){ std::cout << "An unknown error occurred" << std::endl;;}
}
