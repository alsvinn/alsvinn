	
set(MPI_CXX_SKIP_MPICXX ON)
find_package(MPI REQUIRED)
add_definitions(-DALSVINN_USE_MPI)

# It seems MPI doesn't set this correctly anymore, so we have to
# specify This
# manually.
add_definitions(-DOMPI_SKIP_MPICXX)
add_definitions(-DMPICH_SKIP_MPICXX)

if(MPI_mpi_LIBRARY)
    # We set additional include directories, 
    # especially on CSCS daint, this was needed
    get_filename_component(_library_folder ${MPI_mpi_LIBRARY} DIRECTORY)
    get_filename_component(_mpi_folder ${_library_folder} DIRECTORY)
    LIST(APPEND MPI_INCLUDE_DIRS "${_mpi_folder}/include")
    set_property(TARGET MPI::MPI_C PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${MPI_INCLUDE_DIRS})

elseif(MPIEXEC_EXECUTABLE)
    # We set additional include directories, 
    # especially on CSCS daint, this was needed
    get_filename_component(_bin_folder ${MPIEXEC_EXECUTABLE} DIRECTORY)
    get_filename_component(_mpi_folder ${_bin_folder} DIRECTORY)
    LIST(APPEND MPI_INCLUDE_DIRS "${_mpi_folder}/include")
    set_property(TARGET MPI::MPI_C PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${MPI_INCLUDE_DIRS})
endif()

