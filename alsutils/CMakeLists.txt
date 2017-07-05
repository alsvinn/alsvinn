cmake_minimum_required (VERSION 2.8.8)




FILE(GLOB_RECURSE HEADERS include/*.hpp)
FILE(GLOB_RECURSE SRC src/*.cpp)

set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsutils" "${HEADERS}")

ADD_LIBRARY( alsutils SHARED ${SRC} ${HEADERS})

