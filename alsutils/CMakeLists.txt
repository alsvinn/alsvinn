cmake_minimum_required (VERSION 2.8.8)




file(GLOB_RECURSE HEADERS include/*.hpp)
file(GLOB_RECURSE SRC src/*.cpp)

set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsutils" "${HEADERS}")

add_library( alsutils SHARED ${SRC} ${HEADERS})
target_link_libraries(alsutils PUBLIC Boost::log)
target_include_directories(alsutils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      ${CUDA_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>)
