
set(MPI_CXX_SKIP_MPICXX ON)
find_package(MPI REQUIRED)
add_definitions(-DALSVINN_USE_MPI)

# It seems MPI doesn't set this correctly anymore, so we have to
# specify This
# manually.
add_definitions(-DOMPI_SKIP_MPICXX)
add_definitions(-DMPICH_SKIP_MPICXX)

if(NOT MPI_C_INCLUDE_PATH)
        SET(MPI_INCLUDE_PATH_FOUND OFF)
	if(MPI_mpi_LIBRARY)
	    # We set additional include directories,
	    # especially on CSCS daint, this was needed
	    get_filename_component(_library_folder ${MPI_mpi_LIBRARY} DIRECTORY)
	    get_filename_component(_mpi_folder ${_library_folder} DIRECTORY)
	    if(EXISTS "${_mpi_folder}/include")
	        SET(MPI_INCLUDE_PATH_FOUND ON)
	        LIST(APPEND MPI_INCLUDE_DIRS "${_mpi_folder}/include")
	        set_property(TARGET MPI::MPI_C PROPERTY INTERFACE_INCLUDE_DIRECTORIES
	            ${MPI_INCLUDE_DIRS})
	    endif()
	endif()
	if(MPI_C_COMPILER)
	    IF(NOT MPI_INCLUDE_PATH_FOUND)
	        # We set additional include directories,
	        # especially on CSCS daint, this was needed
	        get_filename_component(_bin_folder ${MPI_C_COMPILER} DIRECTORY)
	        get_filename_component(_mpi_folder ${_bin_folder} DIRECTORY)
	        if(EXISTS "${_mpi_folder}/include")
		    SET(MPI_INCLUDE_PATH_FOUND ON)
	            IF(MPI_INCLUDE_DIRS)
	       	        LIST(APPEND MPI_INCLUDE_DIRS "${_mpi_folder}/include")
	            ELSE()
		        SET(MPI_INCLUDE_DIRS "${_mpi_folder}/include")
		    ENDIF()
		ENDIF()
            endif()
	endif()
	if(NOT "$ENV{MPICH_DIR}" STREQUAL "")
	    IF(NOT MPI_INCLUDE_PATH_FOUND)
                if(EXISTS "$ENV{MPICH_DIR}/include")
                    SET(MPI_INCLUDE_PATH_FOUND ON)	
	            IF(MPI_INCLUDE_DIRS)
	       	        LIST(APPEND MPI_INCLUDE_DIRS "$ENV{MPICH_DIR}/include")
	            ELSE()
		        SET(MPI_INCLUDE_DIRS "$ENV{MPICH_DIR}/include")
		    ENDIF()
		endif()
            endif()
	ENDIF()
        if(CMAKE_CXX_COMPILER)
            IF(NOT MPI_INCLUDE_PATH_FOUND)	
	        # We set additional include directories,
	        # especially on CSCS daint, this was needed
	        get_filename_component(_bin_folder ${CMAKE_CXX_COMPILER} DIRECTORY)
	        get_filename_component(_mpi_folder ${_bin_folder} DIRECTORY)
	        if(EXISTS "${_mpi_folder}/include")
	            IF(MPI_INCLUDE_DIRS)
	       	        LIST(APPEND MPI_INCLUDE_DIRS "${_mpi_folder}/include")
	            ELSE()
		        SET(MPI_INCLUDE_DIRS "${_mpi_folder}/include")
		    ENDIF()
                endif()
	    endif()	    
	ENDIF()
	set_property(TARGET MPI::MPI_C PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      	        ${MPI_INCLUDE_DIRS})

endif()

# Again, cuda is acting up, so we need to add a dummy target for only MPI_C
# include directories
add_library(MPI::MPI_C_include INTERFACE IMPORTED)
if (MPI_C_INCLUDE_PATH)
	set_property(TARGET MPI::MPI_C_include PROPERTY INTERFACE_INCLUDE_DIRECTORIES
	${MPI_C_INCLUDE_PATH})
elseif(MPI_INCLUDE_DIRS)
   MESSAGE(${MPI_INCLUDE_DIRS})
	set_property(TARGET MPI::MPI_C_include PROPERTY INTERFACE_INCLUDE_DIRECTORIES
	${MPI_INCLUDE_DIRS})
endif()
