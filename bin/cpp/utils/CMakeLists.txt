MACRO(ADD_PROJECT Name Sources)
    add_executable (${Name} ${Sources})
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

ADD_PROJECT(generate_bin_k_cluster_file generate_bin_k_cluster_file.cpp)

