cmake_minimum_required(VERSION 3.10)
project(dynamic_mosp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

find_package(MPI REQUIRED)
find_library(METIS_LIB metis REQUIRED)

include_directories(${MPI_CXX_INCLUDE_DIRS})
add_executable(dynamic_mosp src/main.cpp)
target_link_libraries(dynamic_mosp ${MPI_CXX_LIBRARIES} ${METIS_LIB})
