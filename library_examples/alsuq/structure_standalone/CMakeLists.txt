cmake_minimum_required (VERSION 3.10.0)

SET(USE_HUNTER OFF CACHE BOOL "Use Hunter for installation of packages")
SET(HUNTER_ROOT ${CMAKE_BINARY_DIR}/hunter_root)

IF(${USE_HUNTER})
	include("cmake/HunterGate.cmake")
	HunterGate(
	    URL "https://github.com/ruslo/hunter/archive/v0.23.123.tar.gz"
            SHA1 "e28cbf2464e922bbeaec0c7d9ad35bc45670435d"
	    LOCAL
	)

ELSE()
	MACRO(hunter_add_package args)
	# do nothing
	ENDMACRO()
ENDIF()
project (structure_functions LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)

if (EXISTS "${CMAKE_SOURCE_DIR}/../../../cmake/git_find/GetGitRevisionDescription.cmake")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmake/git_find")
    include(GetGitRevisionDescription)
    get_git_head_revision(GIT_REFSPEC GIT_SHA1)
    git_local_changes(GIT_VERSION_STATUS)
else()
    SET(GIT_VERSION_STATUS "Unknown")
    SET(GIT_SHA1 "Unknown")
endif()


find_package(Alsvinn REQUIRED)
add_executable(structure_standalone src/main.cpp)



configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/git_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/git_config.hpp ESCAPE_QUOTES)

target_include_directories(structure_standalone PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)


target_link_libraries(structure_standalone
	alsvinn::alsutils
	alsvinn::alsuq
        Boost::log
	Boost::program_options
	Boost::date_time
	Boost::system
	Boost::chrono
	Boost::thread
  
)
