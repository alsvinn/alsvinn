# Downloads the 1.8 version of cub (http://nvlabs.github.io/cub/index.html)

INCLUDE(ExternalProject)

ExternalProject_Add(
    CUB
    URL "https://github.com/NVlabs/cub/archive/v1.8.0.zip"
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CUB
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")

add_library(CUB::CUB_CPP INTERFACE IMPORTED)
set_property(TARGET CUB::CUB_CPP PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/CUB)
set_property(TARGET CUB::CUB_CPP PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_BINARY_DIR}/CUB)
add_dependencies(CUB::CUB_CPP CUB)
