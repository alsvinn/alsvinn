cmake_minimum_required (VERSION 3.1.0)

FILE(GLOB_RECURSE SRC src/*.cpp)
IF(NOT ALSVINN_HAVE_CUDA)
  FILE(GLOB_RECURSE CUDA_SRC src/cuda/*.cpp)
  LIST(REMOVE_ITEM SRC ${CUDA_SRC})
ENDIF()

IF(NOT ALSVINN_HAS_PARALLEL_HDF5)
  FILE(GLOB_RECURSE HDF5_MPI_SRC src/io/HDF5MPIWriter*.cpp)
  LIST(REMOVE_ITEM SRC ${HDF5_MPI_SRC})
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

IF(NOT ALSVINN_HAS_PARALLEL_HDF5)
  FILE(GLOB_RECURSE HDF5_MPI_HEADERS include/alsfvm/io/HDF5MPIWriter*.hpp)
  LIST(REMOVE_ITEM HEADERS ${HDF5_MPI_HEADERS})
ENDIF()  

IF (NOT ALSVINN_USE_MPI)
  FILE(GLOB_RECURSE MPI_HEADERS include/*MPI*.hpp)
  LIST(REMOVE_ITEM HEADERS ${MPI_HEADERS})
ENDIF()

set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsfvm" "${HEADERS}")



IF(ALSVINN_HAVE_CUDA)
  FILE(GLOB_RECURSE CUDA_SRC src/*.cu)
  set_source_groups( "CUDA Files" "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${CUDA_SRC}")
  add_library(alsfvm_cuda SHARED ${CUDA_SRC})
  target_link_libraries(alsfvm_cuda alsutils_include MPI::MPI_C_include Boost::log CUB::CUB_CPP)
  target_include_directories(alsfvm_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)

ELSE()
  add_library(alsfvm_cuda INTERFACE)
ENDIF()

add_library(alsvinn::alsfvm_cuda ALIAS alsfvm_cuda)

# This is a dummy target in a way, but the problem is that the current cmake
# version messes up the cuda flags if we link with alsfvm
# basically, some library exports the -pthreads C flag onto nvcc, and it doesn't
# understand it.
add_library(alsfvm_include INTERFACE)
target_include_directories(alsfvm_include INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_library(alsvinn::alsfvm_include ALIAS alsfvm_include)

add_library( alsfvm SHARED ${SRC} ${HEADERS} )
add_library(alsvinn::alsfvm ALIAS alsfvm)

target_include_directories(alsfvm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(alsfvm
  PUBLIC
  alsutils
  alsfvm_cuda
  Boost::filesystem
  Boost::${ALSVINN_BOOST_PYTHON} 
  Boost::${ALSVINN_BOOST_NUMPY}
  MPI::MPI_C
  Boost::date_time	
  hdf5::HDF5_C
  Python::Python
  ${CUDA_LIBRARIES}
  ${OpenMP_CXX_LIB_NAMES}
  netcdf::NetCDF4_c
  netcdf::parallel_NetCDF4_c
  PRIVATE
  FBM::fbm
  FBM::fbmpy)

install(TARGETS alsfvm_cuda
    EXPORT alsvinn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT alsfvm)

install(TARGETS alsfvm_include
    EXPORT alsvinn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT alsfvm)

install(TARGETS alsfvm
    EXPORT alsvinn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT alsfvm)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT alsfvm)
