cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC src/ *.cpp src/*.hpp)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "../alsfvm/include")
IF(ALSVINN_HAVE_CUDA) 
	CUDA_INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "../alsfvm/include")
	FILE(GLOB_RECURSE CUDA_SRC src/*.cu)
	set_source_groups( "CUDA Files" "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${CUDA_SRC}")
	CUDA_COMPILE(ALSTEST_CUDA alstest_cuda ${CUDA_SRC})
ELSE()
   # Remove files looking like CUDA files
   FILE(GLOB_RECURSE CUDA_FILES src/*CUDA*)
   LIST(REMOVE_ITEM SRC ${CUDA_FILES})
   SET(ALSTEST_CUDA "")
ENDIF()

ADD_EXECUTABLE(alstest ${SRC} ${ALSTEST_CUDA})

ADD_DEPENDENCIES(alstest GTest)

TARGET_LINK_LIBRARIES(alstest ${GTEST_LIBRARIES} alsfvm
                              ${Boost_LIBRARIES}
                              ${CUDA_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
			      ${MPI_LIBRARIES}
                              )
