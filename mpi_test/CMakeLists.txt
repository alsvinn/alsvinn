cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC src/ *.cpp)
FILE(GLOB_RECURSE HEADERS src/ *.hpp)



set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsfvm" "${HEADERS}")

ADD_EXECUTABLE(alstest_mpi ${SRC} ${HEADERS} ${ALSTEST_CUDA})

if(${ALSVINN_USE_MPI})
    SET(ALSUQ_LIBRARY alsuq)
else()
    SET(ALSUQ_LIBRARY "")
endif()

if(ALSVINN_MPI_GPU_DIRECT)
	message(STATUS "Using gpu direct for mpi test")
	# mpich is really strict on this, gpu direct will be disabled
	# o therwise.
	# yes, it's an ugly hack, but it's a working hack.
  SET(CUDARTLIB -lcudart)
ELSE()
	SET(CUDARTLIB "")
endif()
target_include_directories(alstest_mpi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

TARGET_LINK_LIBRARIES(alstest_mpi alsfvm alsuq GTest::GTest GTest::Main)

configure_file(run_mpi_tests.sh.in ${CMAKE_BINARY_DIR}/run_mpi_tests.sh)
