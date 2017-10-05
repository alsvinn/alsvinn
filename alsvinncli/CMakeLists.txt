cmake_minimum_required (VERSION 2.8.8)

FILE(GLOB_RECURSE SRC src/*.cpp)

FILE(GLOB_RECURSE EXAMPLES examples/*.py examples/*.xml)
INCLUDE_DIRECTORIES("../alsfvm/include")
CUDA_ADD_EXECUTABLE(alsvinncli ${SRC} ${EXAMPLES})

if(ALSVINN_USE_CUDA)
	SET(CUDARTLIB "-lcudart")
ELSE()
	SET(CUDARTLIB "")
endif()
TARGET_LINK_LIBRARIES(alsvinncli  ${HDF5_LIBRARIES} ${Boost_LIBRARIES}
                              ${PYTHON_LIBRARY} alsfvm ${HDF5_LIBRARIES}
                              ${Boost_LIBRARIES}
                              ${PYTHON_LIBRARY}
                              ${CUDA_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
			      ${MPI_LIBRARIES}
			      ${CUDARTLIB}
			      )
