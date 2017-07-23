MACRO(ADD_PROJECT Name Sources)
    add_executable (${Name} ${Sources} sgd_kmeans.h mini_batch_kmeans.h math_utils.h)
    target_link_libraries (${Name} pmm)
    target_link_libraries (${Name} LINK_PUBLIC roaring)
    MESSAGE( STATUS "Command line project: " ${Name} )
ENDMACRO()

########################################################################################################

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories ("${CMAKE_SOURCE_DIR}/src/cpp")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(APPLE)
else(APPLE)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
	message (STATUS "OpenMP found")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif(APPLE)

ADD_PROJECT(chunked_memory_iota chunked_memory_iota.cpp)
ADD_PROJECT(chunked_memory_iota_linear chunked_memory_iota_linear.cpp)
ADD_PROJECT(chunked_memory_iota_linear_vec chunked_memory_iota_linear_vec.cpp)
ADD_PROJECT(chunked_memory_random_sqrt_2d chunked_memory_random_sqrt_2d.cpp)
ADD_PROJECT(chunked_memory_random_sqrt_3d chunked_memory_random_sqrt_3d.cpp)
ADD_PROJECT(sgd_kmeans_scan sgd_kmeans_scan.cpp)
ADD_PROJECT(sgd_kmeans_simple sgd_kmeans_simple.cpp)
ADD_PROJECT(sgd_kmeans_chunks sgd_kmeans_chunks.cpp)
ADD_PROJECT(sgd_kmeans_chunks_with_threads sgd_kmeans_chunks_with_threads.cpp)
ADD_PROJECT(mini_batch_kmeans_scan mini_batch_kmeans_scan.cpp)
ADD_PROJECT(mini_batch_kmeans_simple mini_batch_kmeans_simple.cpp)
ADD_PROJECT(mini_batch_kmeans_chunks mini_batch_kmeans_chunks.cpp)
ADD_PROJECT(mini_batch_kmeans_chunks_with_threads mini_batch_kmeans_chunks_with_threads.cpp)

