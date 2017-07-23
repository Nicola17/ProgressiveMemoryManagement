cmake_minimum_required (VERSION 2.8.11)
project (ProgressiveTable)

option(PYTHON_BINDING "Python Binding" OFF)

# Make the scripts available in the 'cmake' directory available for the
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )


add_subdirectory (src/cpp/pmm)
add_subdirectory (bin/cpp/benchmarks)
add_subdirectory (bin/cpp/tdd)
add_subdirectory (bin/cpp/utils)

if(PYTHON_BINDING)
    # Include the CMake script UseCython.cmake.  This defines add_cython_module().
    # Instruction for use can be found at the top of cmake/UseCython.cmake.
    include( UseCython )

    add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
      ${CMAKE_CURRENT_BINARY_DIR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

    add_subdirectory (src/python)
endif(PYTHON_BINDING)
