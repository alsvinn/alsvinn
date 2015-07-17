#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/io/io_utils.hpp"
#include <memory>
#include <cassert>
#include "alsfvm/io/hdf5_utils.hpp"


namespace alsfvm {
namespace io {

HDF5Writer::HDF5Writer(const std::string& basefileName)
    : snapshotNumber(0), basefileName(basefileName)
{
    // empty
}

void HDF5Writer::write(const volume::Volume &conservedVariables,
                       const volume::Volume &extraVariables,
                       const simulator::TimestepInformation &timestepInformation)
{
    std::string name = getOutputname(basefileName, snapshotNumber);
    std::string h5name = name + std::string(".h5");
    HDF5Resource file(H5Fcreate(h5name.c_str(),
                                H5F_ACC_TRUNC, H5P_DEFAULT,
                                H5P_DEFAULT), H5Fclose);


    writeVolume(conservedVariables, file.hid());
    writeVolume(extraVariables, file.hid());
    snapshotNumber++;
}

void HDF5Writer::writeVolume(const volume::Volume &volume, hid_t file)
{
    for(size_t i = 0; i < volume.getNumberOfVariables(); i++)
    {
        writeMemory(*(volume.getScalarMemoryArea(i)), volume.getName(i), file);
    }
}

void HDF5Writer::writeMemory(const memory::Memory<real> &memory,
                             const std::string& name,
                             hid_t file)
{
    //   GROUP "/" {
    //   Group "A" {
    //     Dataset "phi" {
    //       Att vsType = "variable"                     // Required string
    //       Att vsMesh = "mycartgrid"                   // Required string
    //       DATASPACE [200, 300, 104]
    //       Att vsCentering = "zonal"                   // Optional string, defaults to "nodal", other allowed values
    //                                                   // are "zonal", "edge" or "face"
    //       Att vsTimeGroup = "mytime"                  // Optional string
    //     }
    //   }
    // }
    hsize_t dimensions[] = {memory.getSizeX(),
                            memory.getSizeY(),
                            memory.getSizeZ()};

    HDF5Resource filespace(H5Screate_simple(3, dimensions, NULL), H5Sclose);


    HDF5Resource dataset(H5Dcreate(file, name.c_str(), H5T_IEEE_F32LE,
                                   filespace.hid(),
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);



    // Now we will write the portion of data that our process is responsible
    // for (if we are not running MPI, this will default to the whole data)
    // See https://www.hdfgroup.org/HDF5/Tutor/phypecont.html for details on how
    // this works. And also check the C++ example
    // https://www.hdfgroup.org/ftp/HDF5/current/src/unpacked/c++/examples/h5tutr_subset.cpp

    // The number of elements we will write in each direction
    hsize_t count[] = {memory.getSizeX(),
                       memory.getSizeY(),
                       memory.getSizeZ()};

    // The offset (where we will start writing data
    hsize_t offset[] = {0, 0, 0};
    // We need a temporary memory space to hold the data
    HDF5Resource memspace(H5Screate_simple(3, count, NULL), H5Sclose);



    // Last two arguments are NULL since we accept the default value
    HDF5_SAFE_CALL(H5Sselect_hyperslab(filespace.hid(), H5S_SELECT_SET, offset, NULL, count,
            NULL));


    // Here we only support double writing at the moment
    static_assert(std::is_same<real, double>::value, "HDF5 only supports double for now");

    // Need to redo for GPU
    assert(memory.isOnHost());
    // Then we write the data as we normally would.
    HDF5_SAFE_CALL(H5Dwrite(dataset.hid(), H5T_NATIVE_DOUBLE,
                memspace.hid(), filespace.hid(), H5P_DEFAULT,
                memory.getPointer()));

    writeString(dataset.hid(), "vsType", "variable");
    writeString(dataset.hid(), "vsMesh", "grid");
    writeString(dataset.hid(), "vsCentering", "zonal");

}

void HDF5Writer::writeString(hid_t object, const std::string &name, const std::string &value)
{
    HDF5Resource dataspace(H5Screate(H5S_SCALAR), H5Sclose);

    HDF5Resource type(H5Tcopy(H5T_C_S1), H5Tclose);

    HDF5_SAFE_CALL(H5Tset_size(type.hid(), value.size()));

    // Create attribute and write to it
    HDF5Resource attribute(H5Acreate(object, name.c_str(), type.hid(), dataspace.hid(),
                               H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

    const char* cString = value.c_str();
    HDF5_SAFE_CALL(H5Awrite(attribute.hid(), type.hid(), cString));
}

void HDF5Writer::writeFloats(hid_t object, const std::string &name, const std::vector<float> &values)
{

    hsize_t dims = values.size();

    // Create the data space for the attribute.
    HDF5Resource dataspace(H5Screate_simple(1, &dims, NULL), H5Sclose);

    // Create a dataset attribute.
    HDF5Resource attribute(H5Acreate2(object,  name.c_str(), H5T_IEEE_F32LE,
                                 dataspace.hid(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

    HDF5_SAFE_CALL(H5Awrite(attribute.hid(), H5T_NATIVE_INT, values.data()));
}

void HDF5Writer::writeIntegers(hid_t object, const std::string &name, const std::vector<int> &values)
{
    hsize_t dims = values.size();

    // Create the data space for the attribute.
    HDF5Resource dataspace(H5Screate_simple(1, &dims, NULL), H5Sclose);

    // Create a dataset attribute.
    HDF5Resource attribute(H5Acreate2(object,  name.c_str(), H5T_STD_I32LE,
                                 dataspace.hid(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

    HDF5_SAFE_CALL(H5Awrite(attribute.hid(), H5T_NATIVE_INT, values.data()));
}

} // namespace io
} // namespace alsfvm

