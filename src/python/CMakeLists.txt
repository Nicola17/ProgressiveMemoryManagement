# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
  ${CMAKE_SOURCE_DIR}/src/python/pypmm.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

if(APPLE)
else(APPLE)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
	message (STATUS "OpenMP found")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories ("${CMAKE_SOURCE_DIR}/src/cpp")
include_directories ("${CMAKE_SOURCE_DIR}/src/python")


# Multi-file cython modules do not appear to be working at the moment.
cython_add_module(pypmm pypmm.pyx)

target_link_libraries(pypmm pmm)
target_link_libraries(pypmm LINK_PUBLIC roaring)
