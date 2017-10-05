cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC src/ *.cpp)
FILE(GLOB_RECURSE HEADESRS src/ *.hpp)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "../alsfvm/include")
INCLUDE_DIRECTORIES("src")

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
