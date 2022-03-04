cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(MatrixFreePreconditioner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP)



include_directories(external)

add_compile_definitions("DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data/\"")

add_executable(preconditoner_test src/main.cpp)
add_executable(module_test src/test.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(preconditoner_test PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(module_test PUBLIC OpenMP::OpenMP_CXX)
endif()
