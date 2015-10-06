cmake_minimum_required (VERSION 2.8.8)
project (alsvinn)
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

SET(ALSVINN_CXX_FLAGS "" CACHE STRING "CXX flags to use")
SET(ALSVINN_USE_CUDA OFF CACHE BOOL "Build with CUDA")
SET(ALSVINN_BUILD_DOXYGEN ON CACHE BOOL "Build doxygen documentation")
# Add the user's chosen flags to the build for this directory.
if (NOT WIN32)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALSVINN_CXX_FLAGS} -std=c++11 -Wall -Wno-comment")
endif()

IF(WIN32) 
	# This is apparently needed for Visual Studio 2013, could be 
	# fixed on 2013, but cuda doesn't work there...
	ADD_DEFINITIONS(-DBOOST_NO_CXX11_ALLOCATOR)
ENDIF()


if (ALSVINN_USE_CUDA)
    find_package(CUDA REQUIRED)
else()
    set(CUDA_FOUND OFF)
endif()

IF(WIN32)
  ADD_DEFINITIONS( -DBOOST_ALL_NO_LIB )
ENDIF()


# Python dependency could be made optional eventually, but for now it's
# the only way of specify initial data
find_package(PythonLibs REQUIRED)

# We try to limit use of boost, but we also want to avoid having to 
# reprogram everything.
find_package(Boost REQUIRED COMPONENTS filesystem system chrono)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
if(CUDA_FOUND)
	
  IF(NOT WIN32)
	#Mostly only relevant for older versions of CUDA on Linux/Mac OS X, 
	# Basically, we do not want to use std=c++11 on CUDA
	  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  ENDIF()
  #list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code="compute_52,sm_52" -Xptxas -v )
  list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g -O0 -lineinfo -src-in-ptx -Xptxas --device-debug,-O0)
  set(ALSVINN_HAVE_CUDA On)
  add_definitions(-DALSVINN_HAVE_CUDA)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
else()
  set(ALSVINN_HAVE_CUDA Off)
  set(CUDA_INCLUDE_DIRS "")
  set(CUDA_LIBRARIES "")
endif()

find_package(HDF5 REQUIRED)

add_subdirectory("alsfvm")
add_subdirectory("alsvinncli")
add_subdirectory("alsmlmccli")
add_subdirectory("test")
add_subdirectory("system_test")

if(ALSVINN_BUILD_DOXYGEN)
	add_subdirectory("doxygen")
endif()
