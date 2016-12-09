cmake_minimum_required (VERSION 2.8.8)



FILE(GLOB_RECURSE SRC src/*.cpp)
FILE(GLOB_RECURSE HEADERS include/*.hpp)

set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsuq" "${HEADERS}")


INCLUDE_DIRECTORIES("include" ${CUDA_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})


IF(ALSVINN_HAVE_CUDA) 
	CUDA_INCLUDE_DIRECTORIES("include")
	FILE(GLOB_RECURSE CUDA_SRC src/*.cu)
	set_source_groups( "CUDA Files" "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${CUDA_SRC}")
	CUDA_COMPILE(ALSFVM_CUDA alsfvm_cuda ${CUDA_SRC})
	CUDA_COMPILE_PTX(ALSFVM_CUDA_PTX alsfvm_cuda_ptx ${CUDA_SRC} OPTIONS -src-in-ptx)

	# This is so that we can build the ptx files for performance analysis
	add_custom_target(alsfvm_ptx_build ALL
		DEPENDS ${ALSFVM_CUDA_PTX} ${CUDA_SRC}
		SOURCES ${CUDA_SRC})
	set_target_properties(alsfvm_ptx_build PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
ELSE()
	SET(ALSFVM_CUDA "")
ENDIF()

ADD_LIBRARY( alsuq ${SRC} ${HEADERS} ${ALSFVM_CUDA})
TARGET_LINK_LIBRARIES(alsuq alsfvm ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} ${PYTHON_LIBRARY}
${Boost_LIBRARIES})

