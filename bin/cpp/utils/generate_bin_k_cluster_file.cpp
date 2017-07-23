#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <random>

template <typename scalar_type>
void highDimVersor(std::vector<scalar_type>& versor){
    scalar_type length(0);
    for(auto& v : versor){
        v = (rand()%1000)/1000.-0.5;
        length += v*v;
    }
    length = std::sqrt(length);
    for(auto& v : versor){
        v /= length;
    }
}

template <typename scalar_type>
void multiplyVector(std::vector<scalar_type>& vector, scalar_type length){
    for (auto& v : vector){
        v *= length;
    }
}

template <typename scalar_type>
void addVector(std::vector<scalar_type>& vector, const std::vector<scalar_type>& add){
    for(int i = 0; i < vector.size(); ++i){
        vector[i] += add[i];
    }
}


int main(int argc, char *argv[]){
    try{
        typedef float scalar_type;

        if(argc != 5){
            std::cout << "Wrong number of parameters!" << std::endl;
            std::cout << "1: file name" << std::endl;
            std::cout << "2: # dimensions" << std::endl;
            std::cout << "3: # clusters" << std::endl;
            std::cout << "4: # points per cluster" << std::endl;
            return 1;
        }

        const std::string file_name(argv[1]);
        const int n_dims(std::atoi(argv[2]));
        const int n_clusters(std::atoi(argv[3]));
        const int points_per_sphere(std::atoi(argv[4]));
        const scalar_type radius = 1;
        const scalar_type inter_sphere_mult = 100;

        std::default_random_engine generator;
        std::normal_distribution<scalar_type> distribution(0,radius);

        std::ofstream out_file;
        out_file.open(file_name.c_str(), std::ios::out | std::ios::binary);
        for (int i = 0; i < n_clusters; ++i){
            std::vector<scalar_type> origin(n_dims, 0);
            highDimVersor(origin);
            multiplyVector<scalar_type>(origin, distribution(generator)*inter_sphere_mult);

            for (int j = 0; j < points_per_sphere; ++j){
                std::vector<scalar_type> point(origin);
                std::vector<scalar_type> to_add(n_dims);
                highDimVersor(to_add);
                multiplyVector(to_add, distribution(generator));
                addVector(point, to_add);
                out_file.write( (char*)point.data(), sizeof(scalar_type)*n_dims);
            }
        }
        out_file.close();

    }
    catch(std::logic_error& ex){ std::cout << "Logic error: " << ex.what();}
    catch(std::runtime_error& ex){ std::cout << "Runtime error: " << ex.what();}
    catch(...){ std::cout << "An unknown error occurred";}
}
