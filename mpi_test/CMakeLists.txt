cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC src/ *.cpp)
FILE(GLOB_RECURSE HEADESRS src/ *.hpp)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "../alsfvm/include")
INCLUDE_DIRECTORIES("src")

set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsfvm" "${HEADERS}")

CUDA_ADD_EXECUTABLE(alstest_mpi ${SRC} ${HEADERS} ${ALSTEST_CUDA})

if(${ALSVINN_USE_MPI})
    SET(ALSUQ_LIBRARY alsuq)
else()
    SET(ALSUQ_LIBRARY "")
endif()

if(ALSVINN_MPI_GPU_DIRECT)
	message(STATUS "Using gpu direction for mpi test")
	# mpich is really strict on this, gpu direct will be disabled
	# o therwise.
	# yes, it's an ugly hack, but it's a working hack.
        SET(CUDARTLIB -lcudart)
ELSE()
	SET(CUDARTLIB "")
endif()

TARGET_LINK_LIBRARIES(alstest_mpi ${GTEST_LIBRARIES} ${Boost_LIBRARIES} alsfvm ${ALSUQ_LIBRARY} ${HDF5_LIBRARIES}
                              ${Boost_LIBRARIES}
                              ${CUDA_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
                              ${MPI_LIBRARIES}
                              ${NETCDF_LIBRARIES}
			      ${CUDARTLIB}

                              )

configure_file(run_mpi_tests.sh.in ${CMAKE_BINARY_DIR}/run_mpi_tests.sh)
