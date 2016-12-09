cmake_minimum_required (VERSION 3.1.0)
SET(ALSVINN_USE_HUNTER ON CACHE BOOL "Use Hunter for automatic compiling of additional packages")

SET(HUNTER_ROOT ${CMAKE_BINARY_DIR}/hunter_root)

IF(${ALSVINN_USE_HUNTER})
	include("cmake/HunterGate.cmake")

	HunterGate(
	    URL "https://github.com/ruslo/hunter/archive/v0.15.0.tar.gz"
	        SHA1 "5ce8ae8627d94315c47a0acf246de1c0aae2b4b9"
		)
ELSE()
	MACRO(hunter_add_package args)
	     # do nothing
	ENDMACRO()
ENDIF()

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
    # On debug in Windows we want to get floating point execeptions.
    SET(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}  " /fp:strict ")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALSVINN_CXX_FLAGS}  /MP")
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
    ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
ENDIF()
# Python dependency could be made optional eventually, but for now it's
# the only way of specify initial data
find_package(PythonLibs REQUIRED)

# We try to limit use of boost, but we also want to avoid having to 
# reprogram everything.
hunter_add_package(Boost COMPONENTS regex system filesystem chrono date_time program_options)
find_package(Boost REQUIRED filesystem system chrono date_time program_options)


IF(WIN32)
  ADD_DEFINITIONS( -DBOOST_ALL_NO_LIB )
ENDIF()
set(Boost_USE_STATIC_LIBS ON)
if(MSVC)
  add_definitions(-DBOOST_ALL_NO_LIB=1)
endif()

find_package (Threads)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
if(CUDA_FOUND)
	
  #IF(NOT WIN32)
	#Mostly only relevant for older versions of CUDA on Linux/Mac OS X, 
	# Basically, we do not want to use std=c++11 on CUDA
#	  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
 # ENDIF()
  SET(CUDA_COMPILE_PTX ON)
  #list(APPEND CUDA_NVCC_FLAGS -O3 --use_fast_math -gencode=arch=compute_30,code="compute_30,sm_30" -Xptxas -v )
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
SET(ALSVINN_LIBRARY_DIRS "" CACHE STRING "Locations to add to the library directories")
LINK_DIRECTORIES(${ALSVINN_LIBRARY_DIRS})

if(ALSVINN_USE_MPI)
  find_package(MPI REQUIRED)
  add_definitions(-DALSVINN_USE_MPI)
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_DIRS})
  IF (MPI_ROOT)
    INCLUDE_DIRECTORIES(${MPI_ROOT}/include)
  ENDIF()
  INCLUDE_DIRECTORIES(${MPI_C_INCLUDE_PATH})
  IF (NOT MPI_LIBRARIES)
    SET(MPI_LIBRARIES ${MPI_C_LIBRARIES})
  ENDIF()
else()
  set(MPI_LIBRARIES "")
  INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
endif()
include_directories(${CMAKE_SOURCE_DIR}/alsutils/include)
include_directories(${CMAKE_SOURCE_DIR}/alsuq/include)
include_directories(${CMAKE_SOURCE_DIR}/alsfvm/include)
add_subdirectory("alsutils")

add_subdirectory("alsfvm")
add_subdirectory("alsvinncli")
add_subdirectory("alsmlmccli")
add_subdirectory("alsuq")
add_subdirectory("validation")
if(ALSVINN_BUILD_TESTS)
  hunter_add_package(GTest)
  find_package(GTest REQUIRED)
  add_subdirectory("test")
  add_subdirectory("system_test")
endif()

if(ALSVINN_BUILD_DOXYGEN)
	add_subdirectory("doxygen")
endif()
