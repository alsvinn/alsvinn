add_library(fbm INTERFACE)
add_library(fbmpy INTERFACE)
add_library(FBM::fbm ALIAS fbm)
add_library(FBM::fbmpy ALIAS fbmpy)
if (ALSVINN_BUILD_FBM)
    # We don't always have submodules enabled (for instance from github releases)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fractional_brownian_motion/CMakeLists.txt)
       include(ExternalProject)
       ExternalProject_Add(fractional_brownian_motion
         GIT_REPOSITORY https://github.com/alsvinn/fractional_brownian_motion
         CONFIGURE_COMMAND ""
         BUILD_COMMAND ""
         INSTALL_COMMAND ""
         GIT_TAG        v0.1
         SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fractional_brownian_motion
       )
       add_dependencies(fbm fractional_brownian_motion)
       add_dependencies(fbmpy fractional_brownian_motion)
       target_include_directories(fbm INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/fractional_brownian_motion/fbm/include)
       target_include_directories(fbmpy INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/fractional_brownian_motion/fbmpy/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
    else()
        target_include_directories(fbm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/fractional_brownian_motion/fbm/include)
        target_include_directories(fbmpy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/fractional_brownian_motion/fbmpy/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif()
endif()
