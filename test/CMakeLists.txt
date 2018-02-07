cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC src/ *.cpp)
FILE(GLOB_RECURSE HEADESRS src/ *.hpp)

IF (NOT ALSVINN_USE_MPI)
    FILE(GLOB_RECURSE MPI_SRC src/*MPI*.cpp src/uq/*)
    LIST(REMOVE_ITEM SRC ${MPI_SRC})
ENDIF()


set_source_groups( "Source Files" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SRC}")
set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsfvm" "${HEADERS}")

ADD_EXECUTABLE(alstest ${SRC} ${HEADERS} ${ALSTEST_CUDA})

if(${ALSVINN_USE_MPI})
    SET(ALSUQ_LIBRARY alsuq)
else()
    SET(ALSUQ_LIBRARY "")
endif()
target_include_directories(alstest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

TARGET_LINK_LIBRARIES(alstest  alsfvm alsuq GTest::GTest GTest::Main)
