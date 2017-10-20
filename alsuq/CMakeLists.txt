cmake_minimum_required (VERSION 2.8.8)



FILE(GLOB_RECURSE SRC src/*.cpp)
FILE(GLOB_RECURSE HEADERS include/*.hpp)

set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsuq" "${HEADERS}")


INCLUDE_DIRECTORIES("include" ${CUDA_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

IF (NOT ALSVINN_USE_MPI)
    FILE(GLOB_RECURSE MPI_SRC src/*MPI*.cpp src/*mpi*.cpp)
     LIST(REMOVE_ITEM SRC ${MPI_SRC})
ENDIF()

FILE(GLOB_RECURSE HEADERS include/*.hpp)

IF (NOT ALSVINN_USE_MPI)
    FILE(GLOB_RECURSE MPI_HEADERS include/*MPI*.hpp include/*mpi*.hpp)
     LIST(REMOVE_ITEM HEADERS ${MPI_HEADERS})
ENDIF()

IF(ALSVINN_HAVE_CUDA) 
	CUDA_INCLUDE_DIRECTORIES("include")
	FILE(GLOB_RECURSE CUDA_SRC src/*.cu)
	set_source_groups( "CUDA Files" "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${CUDA_SRC}")
        CUDA_ADD_LIBRARY(alsuq_cuda SHARED ${CUDA_SRC})
	CUDA_COMPILE_PTX(ALSUQ_CUDA_PTX alsuq_cuda_ptx ${CUDA_SRC} OPTIONS -src-in-ptx)

	# This is so that we can build the ptx files for performance analysis
	add_custom_target(alsuq_ptx_build ALL
		DEPENDS ${ALSUQ_CUDA_PTX} ${CUDA_SRC}
		SOURCES ${CUDA_SRC})
	set_target_properties(alsuq_ptx_build PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
	SET(ALSUQ_CUDA alsuq_cuda)
ELSE()
	SET(ALSUQ_CUDA "")
ENDIF()

ADD_LIBRARY( alsuq SHARED ${SRC} ${HEADERS})
TARGET_LINK_LIBRARIES(alsuq alsfvm alsutils  ${ALSUQ_CUDA} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} ${PYTHON_LIBRARY} ${MPI_LIBRARIES})

