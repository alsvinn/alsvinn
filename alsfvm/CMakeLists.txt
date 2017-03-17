cmake_minimum_required (VERSION 2.8.8)



FILE(GLOB_RECURSE SRC src/*.cpp)
IF(NOT ALSVINN_HAVE_CUDA)
    FILE(GLOB_RECURSE CUDA_SRC src/cuda/*.cpp)
    LIST(REMOVE_ITEM SRC ${CUDA_SRC})
ENDIF()

IF (NOT ALSVINN_USE_MPI)
    FILE(GLOB_RECURSE MPI_SRC src/*MPI*.cpp)
     LIST(REMOVE_ITEM SRC ${MPI_SRC})
ENDIF()

FILE(GLOB_RECURSE HEADERS include/*.hpp)

IF(NOT ALSVINN_HAVE_CUDA)
    FILE(GLOB_RECURSE CUDA_HEADERS include/alsfvm/cuda/*.hpp)
    LIST(REMOVE_ITEM HEADERS ${CUDA_HEADERS})
ENDIF()

IF (NOT ALSVINN_USE_MPI)
    FILE(GLOB_RECURSE MPI_HEADERS include/*MPI*.hpp)
     LIST(REMOVE_ITEM HEADERS ${MPI_HEADERS})
ENDIF()

set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsfvm" "${HEADERS}")


INCLUDE_DIRECTORIES("include" ${CUDA_INCLUDE_DIRS}  ${PYTHON_INCLUDE_DIRS})


IF(ALSVINN_HAVE_CUDA) 

	CUDA_INCLUDE_DIRECTORIES("include")
	FILE(GLOB_RECURSE CUDA_SRC src/*.cu)
	set_source_groups( "CUDA Files" "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${CUDA_SRC}")
        CUDA_ADD_LIBRARY(alsfvm_cuda SHARED ${CUDA_SRC})
	CUDA_COMPILE_PTX(ALSFVM_CUDA_PTX alsfvm_cuda_ptx ${CUDA_SRC} OPTIONS -src-in-ptx)

	# This is so that we can build the ptx files for performance analysis
	add_custom_target(alsfvm_ptx_build ALL
		DEPENDS ${ALSFVM_CUDA_PTX} ${CUDA_SRC}
		SOURCES ${CUDA_SRC})
	set_target_properties(alsfvm_ptx_build PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
ELSE()
	SET(ALSFVM_CUDA "")
ENDIF()

ADD_LIBRARY( alsfvm SHARED ${SRC} ${HEADERS} )
TARGET_LINK_LIBRARIES(alsfvm alsfvm_cuda ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} ${PYTHON_LIBRARY} ${NETCDF_LIBRARIES})

