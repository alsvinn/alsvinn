cmake_minimum_required (VERSION 2.8.8)

FILE(GLOB_RECURSE SRC src/*.cpp)

FILE(GLOB_RECURSE EXAMPLES examples/*.py examples/*.xml)

ADD_EXECUTABLE(alsuqcli ${SRC} ${EXAMPLES})

TARGET_LINK_LIBRARIES(alsuqcli ${Boost_LIBRARIES}  alsfvm alsuq
                               ${Boost_LIBRARIES}
                              ${CUDA_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
			      ${MPI_LIBRARIES}
			      )
