cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC *.py)


configure_file(alsvinn/config.py.in ${CMAKE_BINARY_DIR}/python/alsvinn/config.py)
configure_file(alsvinn/config_install.py.in ${CMAKE_BINARY_DIR}/python_install/alsvinn/config.py)


file(COPY ${SRC} DESTINATION ${CMAKE_BINARY_DIR}/python/alsvinn/)

install(FILES ${SRC} DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${ALSVINN_PYTHON_VERSION}/site-packages/alsvinn/)
install(FILES ${CMAKE_BINARY_DIR}/python_install/alsvinn/config.py DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${ALSVINN_PYTHON_VERSION}/site-packages/alsvinn/)
