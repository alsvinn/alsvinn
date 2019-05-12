cmake_minimum_required (VERSION 3.12.0)

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
project (generate_parameters LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
find_package(Alsvinn REQUIRED)
add_executable(generate_parameters src/main.cpp)

target_link_libraries(generate_parameters alsvinn::alsuq)
