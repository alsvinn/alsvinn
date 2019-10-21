
include(GenerateExportHeader)

SET(QMC_GENERATORS_INCLUDE_DIRS "")

SET(QMC_SOURCES "")
SET(QMC_LIBRARY_NAMES "")
if (ALSVINN_BUILD_QMC)
    # We don't always have submodules enabled (for instance from github releases)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/CMakeLists.txt)
       include(ExternalProject)
       ExternalProject_Add(qmc_generators
         GIT_REPOSITORY https://github.com/alsvinn/qmc_generators
         CONFIGURE_COMMAND ""
         BUILD_COMMAND ""
         INSTALL_COMMAND ""
         #GIT_TAG        v0.1
         SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/qmc_generators
       )
    else()
        add_custom_target( qmc_generators )
    endif()


SET(QMC_GENERATORS sobol halton faure halton409 hammersley latin_random)


foreach(QMCDistributionName IN LISTS QMC_GENERATORS)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/alsuq/addons/qmc_generators/QMCBaseTemplate.hpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/alsuq/addons/qmc_generators/QMCDistribution${QMCDistributionName}.hpp ESCAPE_QUOTES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/alsuq/addons/qmc_generators/QMCBaseTemplate.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/src/alsuq/addons/qmc_generators/QMCDistribution${QMCDistributionName}.cpp ESCAPE_QUOTES)

    LIST(APPEND QMC_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/src/alsuq/addons/qmc_generators/QMCDistribution${QMCDistributionName}.cpp)
    LIST(APPEND QMC_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/include/alsuq/addons/qmc_generators/QMCDistribution${QMCDistributionName}.hpp)

    LIST(APPEND QMC_GENERATORS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/${QMCDistributionName}/include)
    LIST(APPEND QMC_GENERATORS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/${QMCDistributionName}/include/qmc)

    SET(QMC_LIBRARY_NAME alsvinn_qmc_${QMCDistributionName})
    add_library(${QMC_LIBRARY_NAME} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/${QMCDistributionName}/src/${QMCDistributionName}.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/${QMCDistributionName}/src/${QMCDistributionName}_api.cpp
        )

    string(TOUPPER ${QMCDistributionName} QMCDistributionNameUpper )
    GENERATE_EXPORT_HEADER(${QMC_LIBRARY_NAME}
                     BASE_NAME ${QMCDistributionName}
                     EXPORT_MACRO_NAME ${QMCDistributionNameUpper}_EXPORT
                     EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/${QMCDistributionName}/include/qmc/${QMCDistributionName}_export.h
    )
    target_include_directories(${QMC_LIBRARY_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/${QMCDistributionName}/include/qmc
        ${CMAKE_CURRENT_SOURCE_DIR}/qmc_generators/${QMCDistributionName}/include)
    LIST(APPEND QMC_LIBRARY_NAMES ${QMC_LIBRARY_NAME})

    install(TARGETS ${QMC_LIBRARY_NAME}
        EXPORT alsvinn-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT qmc_base)
endforeach()
else()
    SET(QMC_GENERATORS sobol halton)
endif()

add_library(qmc_base SHARED
        src/alsuq/addons/qmc_generators/QMCDistribution.cpp
        src/alsuq/addons/qmc_generators/QMCFactory.cpp
        include/alsuq/addons/qmc_generators/QMCDistribution.hpp
        include/alsuq/addons/qmc_generators/QMCFactory.hpp
        ${QMC_SOURCES})
add_dependencies(qmc_base qmc_generators)
add_library(QMC::qmc_base ALIAS qmc_base)

target_include_directories(qmc_base PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE ${QMC_GENERATORS_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/)

target_link_libraries(qmc_base alsutils ${QMC_LIBRARY_NAMES})
        

install(TARGETS qmc_base
    EXPORT alsvinn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT qmc_base)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT qmc_base)
