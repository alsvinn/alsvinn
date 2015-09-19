cmake_minimum_required (VERSION 2.8.8)
project (alsvinn)
SET(ALSVINN_CXX_FLAGS "" CACHE STRING "CXX flags to use")
SET(ALSVINN_USE_CUDA OFF CACHE BOOL "Build with CUDA")
# Add the user's chosen flags to the build for this directory.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALSVINN_CXX_FLAGS} -std=c++11")


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
find_package(Boost REQUIRED COMPONENTS filesystem system)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
if(CUDA_FOUND)
  set(ALSVINN_HAVE_CUDA On)
  add_definitions(-DALSVINN_HAVE_CUDA)
else()
  set(ALSVINN_HAVE_CUDA Off)
  set(CUDA_INCLUDE_DIRS "")
  set(CUDA_LIBRARIES "")
endif()

find_package(HDF5 REQUIRED)

add_subdirectory("alsfvm")
add_subdirectory("alsvinncli")
add_subdirectory("test")
add_subdirectory("system_test")
add_subdirectory("doxygen")
