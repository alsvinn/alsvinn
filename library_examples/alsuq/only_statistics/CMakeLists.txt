cmake_minimum_required (VERSION 3.4.0)
project (using_alsuq LANGUAGES C CXX CUDA)
find_package(Alsvinn REQUIRED)
add_executable(using_alsuq src/main.cpp)

target_link_libraries(using_alsuq alsvinn::alsuq)
