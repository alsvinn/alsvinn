cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC *.py)


configure_file(alsvinn/config.py.in ${CMAKE_BINARY_DIR}/python/alsvinn/config.py)
file(COPY ${SRC} DESTINATION ${CMAKE_BINARY_DIR}/python/alsvinn/)
