cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC src/ *.cpp)
FILE(GLOB_RECURSE HEADESRS src/ *.hpp)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "../alsfvm/include")
INCLUDE_DIRECTORIES("src")
IF(ALSVINN_HAVE_CUDA) 
	CUDA_INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "../alsfvm/include")
	FILE(GLOB_RECURSE CUDA_SRC src/*.cu)
	set_source_groups( "CUDA Files" "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${CUDA_SRC}")
	CUDA_COMPILE(ALSTEST_CUDA alstest_cuda ${CUDA_SRC})
ELSE()
   # Remove files looking like CUDA files
   FILE(GLOB_RECURSE CUDA_FILES src/*CUDA*)
   LIST(REMOVE_ITEM SRC ${CUDA_FILES})
   SET(ALSTEST_CUDA "")
ENDIF()

IF (NOT ALSVINN_USE_MPI)
    FILE(GLOB_RECURSE MPI_SRC src/*MPI*.cpp src/uq/*)
    LIST(REMOVE_ITEM SRC ${MPI_SRC})
ENDIF()


set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsfvm" "${HEADERS}")

ADD_EXECUTABLE(alstest_mpi ${SRC} ${HEADERS} ${ALSTEST_CUDA})

if(${ALSVINN_USE_MPI})
    SET(ALSUQ_LIBRARY alsuq)
else()
    SET(ALSUQ_LIBRARY "")
endif()

TARGET_LINK_LIBRARIES(alstest_mpi ${GTEST_LIBRARIES} ${Boost_LIBRARIES} alsfvm ${ALSUQ_LIBRARY} ${HDF5_LIBRARIES}
                              ${Boost_LIBRARIES}
                              ${CUDA_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
                              ${MPI_LIBRARIES}
                              ${NETCDF_LIBRARIES}

                              )

configure_file(run_mpi_tests.sh.in ${CMAKE_BINARY_DIR}/run_mpi_tests.sh)
