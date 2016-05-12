configure_file(run_validation.sh.in ${CMAKE_BINARY_DIR}/run_validation.sh)

# Add files to make them included in the IDE:
FILE(GLOB_RECURSE ${VALIDATION_FILES} . *.py)

