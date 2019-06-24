CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(dll_writer_example C CXX)
include(GenerateExportHeader)
find_package(MPI REQUIRED)
set(CMAKE_CXX_STANDARD 14)

add_library(dll_writer_example SHARED src/dll_writer_example.cpp)
target_compile_definitions(dll_writer_example PUBLIC -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
GENERATE_EXPORT_HEADER( dll_writer_example
             BASE_NAME dll_writer_example
             EXPORT_MACRO_NAME DLL_WRITER_EXPORT
             EXPORT_FILE_NAME dll_writer_example_exports.h
)
target_link_libraries(dll_writer_example MPI::MPI_C)
target_include_directories(dll_writer_example PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/qmc>
    )
