cmake_minimum_required (VERSION 3.10.0)

SET(USE_HUNTER ON CACHE BOOL "Use Hunter for installation of packages")
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

find_package(Alsvinn REQUIRED)
add_executable(structure_functions src/main.cpp)

target_link_libraries(structure_functions
	alsvinn::alsutils
	alsvinn::alsuq
	Boost::program_options
	Boost::date_time
	Boost::system
	Boost::chrono
	Boost::thread
  
)
