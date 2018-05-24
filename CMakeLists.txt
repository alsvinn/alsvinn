cmake_minimum_required (VERSION 3.1.0)

SET(ALSVINN_USE_FLOAT OFF CACHE BOOL "Do all computations with 32 bits
floating point numbers, instead of 64 bit double precision")
SET(ALSVINN_USE_CUDA ON CACHE BOOL "Build with CUDA")
SET(ALSVINN_USE_HUNTER OFF CACHE BOOL "Use Hunter for automatic compiling of additional packages")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

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
if(${ALSVINN_USE_CUDA})
	project (alsvinn C CXX CUDA)

	# We still need the libraries IMPORTED
	find_package(CUDA REQUIRED)
else()
	project (alsvinn C CXX)
endif()
INCLUDE("cmake/version.cmake")      
ADD_DEFINITIONS(-DCOMPILING_ALSVINN)

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

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/Modules)
SET(ALSVINN_BUILD_TESTS ON CACHE BOOL "Build unittests (requires Gtest). HIGHLY recommended!")
SET(ALSVINN_USE_MPI ON CACHE BOOL "Build with MPI")
SET(ALSVINN_CXX_FLAGS "" CACHE STRING "CXX flags to use")

SET(ALSVINN_BUILD_DOXYGEN ON CACHE BOOL "Build doxygen documentation")
SET(ALSVINN_PROFILE OFF CACHE BOOL "Add profiling information")
SET(ALSVINN_MPI_GPU_DIRECT OFF CACHE BOOL "Use NVidia's GPU direct to do MPI transfers (this is faster, but not only available)")

ADD_DEFINITIONS(-DMPICH_RDMA_ENABLED_CUDA=1)
FIND_PACKAGE(OpenMP REQUIRED)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-undefined-var-template>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-gnu-zero-variadic-macro-arguments>)
endif()

if (NOT WIN32)
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-comment>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

else()
	# This is apparently needed for Visual Studio 2013, could be
	# fixed on 2015, but cuda doesn't work there...
	ADD_DEFINITIONS(-DBOOST_NO_CXX11_ALLOCATOR)

	ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
endif()

# Python dependency could be made optional eventually, but for now it's
# the only way of specify initial data
find_package(PythonLibs REQUIRED)


hunter_add_package(Boost COMPONENTS regex thread system filesystem chrono date_time program_options log)
find_package(Boost REQUIRED filesystem thread system chrono date_time program_options log log_setup python numpy)


if (NOT ALSVINN_USE_HUNTER)
	add_definitions(-DBOOST_LOG_DYN_LINK)
endif()

IF(WIN32)
	add_definitions(-DBOOST_ALL_NO_LIB=1)
endif()

find_package (Threads)
include("cmake/hdf5.cmake")
include("cmake/netcdf.cmake")
include("cmake/python.cmake")
if (ALSVINN_USE_CUDA)
    include("cmake/cub.cmake")
endif()

if(ALSVINN_USE_CUDA)
        set(CMAKE_CUDA_FLAGS "--use_fast_math -Xptxas -warn-spills -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61" )
	set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -lineinfo -src-in-ptx -Xptxas --device-debug,-O0")
	set(ALSVINN_HAVE_CUDA On)
	add_definitions(-DALSVINN_HAVE_CUDA)
endif()

if(ALSVINN_USE_FLOAT)
	ADD_DEFINITIONS(-DALSVINN_USE_FLOAT)
endif()

find_package(HDF5 REQUIRED)


FIND_PACKAGE(NetCDF REQUIRED)


if(ALSVINN_USE_MPI)
    include("cmake/mpi.cmake")
endif()


add_subdirectory("alsutils")

add_subdirectory("alsfvm")
add_subdirectory("alsvinncli")

IF(ALSVINN_USE_MPI)
	add_subdirectory("alsuq")
	add_subdirectory("alsuqcli")
ENDIF()
if(ALSVINN_BUILD_TESTS)
	hunter_add_package(GTest)
	find_package(GTest REQUIRED)
	add_subdirectory("test")
	add_subdirectory("system_test")
	if(ALSVINN_USE_MPI)
		add_subdirectory("mpi_test")
	endif()

endif()

if(ALSVINN_BUILD_DOXYGEN)
	add_subdirectory("doxygen")
endif()
add_subdirectory("python")
