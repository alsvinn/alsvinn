cmake_minimum_required (VERSION 2.8.8)

FILE(GLOB_RECURSE SRC src/*.cpp)

FILE(GLOB_RECURSE EXAMPLES examples/*.py examples/*.xml)
INCLUDE_DIRECTORIES("../alsfvm/include")
ADD_EXECUTABLE(alsvinncli ${SRC} ${EXAMPLES})
TARGET_LINK_LIBRARIES(alsvinncli alsfvm
                              ${CUDA_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT})
