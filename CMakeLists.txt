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
SET(ALSVINN_BUILD_TESTS ON CACHE BOOL "Build unittests (requires Gtest). HIGHLY recommended!")
SET(ALSVINN_USE_MPI OFF CACHE BOOL "Build with MPI")
SET(ALSVINN_CXX_FLAGS "" CACHE STRING "CXX flags to use")
SET(ALSVINN_USE_CUDA ON CACHE BOOL "Build with CUDA")
SET(ALSVINN_BUILD_DOXYGEN ON CACHE BOOL "Build doxygen documentation")
# Add the user's chosen flags to the build for this directory.
set (CMAKE_CXX_STANDARD 11)
if (NOT WIN32)
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALSVINN_CXX_FLAGS} -std=c++11 -Wall -Wno-comment -fopenmp")
else()
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALSVINN_CXX_FLAGS}")
endif()

IF(WIN32) 
	# This is apparently needed for Visual Studio 2013, could be 
        # fixed on 2015, but cuda doesn't work there...
	ADD_DEFINITIONS(-DBOOST_NO_CXX11_ALLOCATOR)
ENDIF()


if (ALSVINN_USE_CUDA)
    find_package(CUDA REQUIRED)
else()
    set(CUDA_FOUND OFF)
endif()

IF(WIN32)
  ADD_DEFINITIONS( -DBOOST_ALL_NO_LIB )
  ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK )
ENDIF()


# Python dependency could be made optional eventually, but for now it's
# the only way of specify initial data
find_package(PythonLibs REQUIRED)

# We try to limit use of boost, but we also want to avoid having to 
# reprogram everything.
find_package(Boost REQUIRED COMPONENTS filesystem system chrono date_time program_options)
find_package (Threads)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
if(CUDA_FOUND)
	
  #IF(NOT WIN32)
	#Mostly only relevant for older versions of CUDA on Linux/Mac OS X, 
	# Basically, we do not want to use std=c++11 on CUDA
#	  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
 # ENDIF()
  SET(CUDA_COMPILE_PTX ON)
  #list(APPEND CUDA_NVCC_FLAGS -O3 --use_fast_math -gencode=arch=compute_20,code="compute_20,sm_20" -Xptxas -v )
  list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3 --use_fast_math -Xptxas -v -keep  -src-in-ptx -gencode=arch=compute_30,code="compute_30,sm_30")
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

if(ALSVINN_USE_MPI)
  find_package(MPI REQUIRED)
else()
  set(MPI_LIBRARIES "")
endif()

add_subdirectory("alsfvm")
add_subdirectory("alsvinncli")
add_subdirectory("alsmlmccli")
add_subdirectory("validation")
if(ALSVINN_BUILD_TESTS)
  add_subdirectory("test")
  add_subdirectory("system_test")
endif()

if(ALSVINN_BUILD_DOXYGEN)
	add_subdirectory("doxygen")
endif()
