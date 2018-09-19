SET(HDF5_NO_FIND_PACKAGE_CONFIG_FILE ON)
find_package(HDF5 REQUIRED)

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
