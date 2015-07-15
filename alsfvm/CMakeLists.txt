cmake_minimum_required (VERSION 2.8.11)

MACRO(set_source_groups  folder_name directory source_files)
	foreach(f ${source_files})
		# Get the path of the file relative to ${DIRECTORY},
		# then alter it (not compulsory)

		file(RELATIVE_PATH SRCGR ${directory} ${f})
		set(SRCGR "${folder_name}/${SRCGR}")
		
		# Extract the folder, ie remove the filename part
		string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})

		# Source_group expects \\ (double antislash), not / (slash)
		string(REPLACE / \\ SRCGR ${SRCGR})
		source_group("${SRCGR}" FILES ${f})
	endforeach()
ENDMACRO(set_source_groups)


FILE(GLOB_RECURSE SRC src/*.cpp)
IF(NOT ALSVINN_HAVE_CUDA)
    FILE(GLOB_RECURSE CUDA_SRC src/cuda/*.cpp)
    LIST(REMOVE_ITEM SRC ${CUDA_SRC})
ENDIF()

FILE(GLOB_RECURSE HEADERS include/*.hpp)

IF(NOT ALSVINN_HAVE_CUDA)
    FILE(GLOB_RECURSE CUDA_HEADERS include/alsfvm/cuda/*.hpp)
    LIST(REMOVE_ITEM HEADERS ${CUDA_HEADERS})
ENDIF()

set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsfvm" "${HEADERS}")


INCLUDE_DIRECTORIES("include" ${CUDA_INCLUDE_DIRS})


IF(ALSVINN_HAVE_CUDA) 
	CUDA_INCLUDE_DIRECTORIES("include")
	FILE(GLOB_RECURSE CUDA_SRC src/*.cu)
	set_source_groups( "CUDA Files" "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${CUDA_SRC}")
	CUDA_COMPILE(ALSFVM_CUDA alsfvm_cuda ${CUDA_SRC})
ELSE()
	SET(ALSFVM_CUDA "")
ENDIF()

ADD_LIBRARY(alsfvm ${SRC} ${HEADERS} ${ALSFVM_CUDA})
TARGET_LINK_LIBRARIES(alsfvm ${CUDA_LIBRARIES})

