cmake_minimum_required (VERSION 3.1)




file(GLOB_RECURSE HEADERS include/*.hpp)
file(GLOB_RECURSE SRC src/*.cpp)

set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsutils" "${HEADERS}")

add_library( alsutils SHARED ${SRC} ${HEADERS})
target_link_libraries(alsutils PUBLIC Boost::log)
target_include_directories(alsutils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
      ${CUDA_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>)

# Dump all variables to the header file
SET(ALL_CONFIGURATIONS "")
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    SET(ALL_CONFIGURATIONS "${ALL_CONFIGURATIONS}\n ${_variableName}=${${_variableName}}")
endforeach()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/alsutils/config.hpp.in
${CMAKE_CURRENT_BINARY_DIR}/include/alsutils/config.hpp ESCAPE_QUOTES)
