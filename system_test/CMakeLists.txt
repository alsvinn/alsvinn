cmake_minimum_required (VERSION 2.8.8)

# We need the thread package for GTest
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(GTest REQUIRED)

FILE(GLOB_RECURSE SRC src/*.cpp)

ADD_EXECUTABLE(alssystemtest ${SRC})
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "../alsfvm/include")

TARGET_LINK_LIBRARIES(alssystemtest ${GTEST_LIBRARIES} alsfvm
                              ${CUDA_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
                              )
