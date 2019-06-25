cmake_minimum_required (VERSION 3.1)




file(GLOB_RECURSE HEADERS include/*.hpp)
file(GLOB_RECURSE SRC src/*.cpp)

set_source_groups( "Header Files" "${CMAKE_CURRENT_SOURCE_DIR}/include/alsutils" "${HEADERS}")


# This is a dummy target in a way, but the problem is that the current cmake
# version messes up the cuda flags if we link with alsfvm
# basically, some library exports the -pthreads C flag onto nvcc, and it
# doesn't
# understand it.
add_library(alsutils_include INTERFACE)
add_library(alsvinn::alsutils_include ALIAS alsutils_include)
target_include_directories(alsutils_include INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/config.cpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.cpp ESCAPE_QUOTES)

add_library( alsutils SHARED ${SRC} ${HEADERS}  ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
add_library(alsvinn::alsutils ALIAS alsutils)
target_link_libraries(alsutils PUBLIC Boost::log  Boost::date_time MPI::MPI_C ${CUDA_LIBRARIES} Python::Python)


target_include_directories(alsutils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
      ${CUDA_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>)

if (NOT ALSVIN_USE_HUNTER)
   target_compile_definitions(alsutils PUBLIC
       -DCOMPILING_ALSVINN
       -DBOOST_LOG_DYN_LINK)
else()
   target_compile_definitions(alsutils PUBLIC
       -DCOMPILING_ALSVINN)
endif()

target_compile_features(alsutils PUBLIC cxx_std_14)

install(TARGETS alsutils_include
    EXPORT alsvinn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT alsutils)

install(TARGETS alsutils
    EXPORT alsvinn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT alsutils)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT alsutils)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/alsutils/config.hpp.in
${CMAKE_CURRENT_BINARY_DIR}/include/alsutils/config.hpp ESCAPE_QUOTES)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/alsutils/config.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alsutils/)
