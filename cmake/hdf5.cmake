SET(HDF5_NO_FIND_PACKAGE_CONFIG_FILE ON)
if (ALSVINN_USE_HUNTER)
   hunter_add_package(hdf5)
   find_package(ZLIB CONFIG REQUIRED)
   find_package(szip CONFIG REQUIRED)
   find_package(hdf5 CONFIG REQUIRED)	
else()
    find_package(HDF5 REQUIRED)
endif()		  

# This gets a bit complicated, but basically, if
# the compile wrapper includes HDF5, we only want to
# export a dummy target, otherwise, we made the imported target

if(HDF5_INCLUDE_DIRS)
    add_library(hdf5::HDF5_C SHARED IMPORTED)

    set_property(TARGET hdf5::HDF5_C PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${HDF5_INCLUDE_DIRS})


    set_property(TARGET hdf5::HDF5_C PROPERTY IMPORTED_LOCATION
        ${HDF5_C_LIBRARY_hdf5})
    
    set_property(TARGET hdf5::HDF5_C PROPERTY
    	IMPORTED_LINK_INTERFACE_LIBRARIES ${HDF5_C_LIBRARIES})
elseif(HDF5_INCLUDE_DIR) 
    add_library(hdf5::HDF5_C SHARED IMPORTED)
    message(${HDF5_INCLUDE_DIR})
    set_property(TARGET hdf5::HDF5_C PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${HDF5_INCLUDE_DIR})

    if(HDF5_C_LIBRARY_hdf5)
        set_property(TARGET hdf5::HDF5_C PROPERTY IMPORTED_LOCATION
    	    ${HDF5_C_LIBRARY_hdf5})
    elseif(HDF5_hdf5_LIBRARY_RELEASE)
        set_property(TARGET hdf5::HDF5_C PROPERTY IMPORTED_LOCATION
    	    ${HDF5_hdf5_LIBRARY_RELEASE})
    endif()

else()
    # Dummy library
    add_library(hdf5::HDF5_C INTERFACE IMPORTED)
endif()

if(HDF5_IS_PARALLEL)
   SET(ALSVINN_HAS_PARALLEL_HDF5 ON)
else()
   SET(ALSVINN_HAS_PARALLEL_HDF5 OFF)
endif()