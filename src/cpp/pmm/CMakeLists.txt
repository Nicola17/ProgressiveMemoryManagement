set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories ("${CMAKE_SOURCE_DIR}/src/cpp")

if(APPLE)
else(APPLE)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        message (STATUS "OpenMP found")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif(APPLE)

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library   (pmm ${HeaderFiles} ${SourceFiles})

set_property(TARGET pmm PROPERTY POSITION_INDEPENDENT_CODE TRUE)

########### INSTALL ##############
install (TARGETS pmm DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/pmm)
