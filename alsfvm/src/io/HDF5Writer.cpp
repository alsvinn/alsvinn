#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/io/io_utils.hpp"
#include <memory>
#include <cassert>
#include "alsfvm/io/hdf5_utils.hpp"
#include "alsfvm/io/io_utils.hpp"
#include <mutex>

// It seems like HDF5 doesn't like to be accessed from different threads...
static std::mutex mutex;
namespace alsfvm {
namespace io {

HDF5Writer::HDF5Writer(const std::string& basefileName)
    : snapshotNumber(0), basefileName(basefileName)
{
    // empty
}

void HDF5Writer::write(const volume::Volume &conservedVariables,
                       const volume::Volume &extraVariables,
					   const grid::Grid& grid,
                       const simulator::TimestepInformation &timestepInformation)
{

    // for hdf5, often the version we use is not thread safe.
    std::unique_lock<std::mutex> lock(mutex);
    std::string name = getOutputname(basefileName, snapshotNumber);
    std::string h5name = name + std::string(".h5");
    HDF5Resource file(H5Fcreate(h5name.c_str(),
                                H5F_ACC_TRUNC, H5P_DEFAULT,
                                H5P_DEFAULT), H5Fclose);

	writeGrid(file.hid(), grid);
    writeVolume(conservedVariables, file.hid());
    writeVolume(extraVariables, file.hid());
    snapshotNumber++;
}

void HDF5Writer::writeGrid(hid_t object, const grid::Grid& grid) {
	HDF5Resource gridGroup(H5Gcreate2(object, "grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
	
	// The comment below is from https://ice.txcorp.com/trac/vizschema/wiki/WikiStart
	// Describe the mesh:
	//   Group "mycartgrid" {
	//   Att vsType = "mesh"                         // Required string
	//   Att vsKind = "uniform"                      // Required string
	//   Att vsStartCell = [0, 0, 0]                 // Required integer array if part of a larger mesh
	//   Att vsNumCells = [200, 200, 104]            // Required integer array giving the number of cells in the x, y, and z directions, respectively
	//   Att vsIndexOrder = "compMinorC"             // Default value is "compMinorC", with the other choice being "compMinorF".
	//                                               // ("compMajorC" and "compMajorF" have the same result as the minor variants). 
	//   Att vsLowerBounds = [-2.5, -2.5, -1.3]      // Required float array
	//   Att vsUpperBounds = [2.5, 2.5, 1.3]         // Required float array
	//   Att vsTemporalDimension = 0                 // Optional unsigned integer denoting which axis is time.
	//                                               // No temporal axis if this attribute is not present.
	// }
	writeString(gridGroup.hid(), "vsType", "mesh");
	writeString(gridGroup.hid(), "vsKind", "uniform");
    writeString(gridGroup.hid(), "vsIndexOrder", "compMinorF");

	writeIntegers(gridGroup.hid(), "vsStartCell", { 0, 0, 0 });
	writeIntegers(gridGroup.hid(), "vsNumCells", grid.getDimensions().toStdVector());
	writeFloats(gridGroup.hid(), "vsLowerBounds", 
		grid.getOrigin().convert<float>().toStdVector());
	writeFloats(gridGroup.hid(), "vsUpperBounds", 
		grid.getTop().convert<float>().toStdVector());
    
}

void HDF5Writer::writeTimeGroup(hid_t object, const simulator::TimestepInformation &timestepInformation)
{
    
}

void HDF5Writer::writeVolume(const volume::Volume &volume, hid_t file)
{
    for(size_t i = 0; i < volume.getNumberOfVariables(); i++)
    {
        writeMemory(volume, i, volume.getName(i), file);
    }
}

///
/// \brief createDatasetForMemroy creates a dataset for the given memory
/// \param volume the volume to read from
/// \param index the index of the memory area to read from
/// \param name the name of the memory (variable name)
/// \param file the file to write to
///
hid_t HDF5Writer::createDatasetForMemory(const volume::Volume& volume, size_t index, const std::string& name,
                 hid_t file) {
    // The comment below is from https://ice.txcorp.com/trac/vizschema/wiki/WikiStart
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
    hsize_t dimensions[] = {volume.getNumberOfXCells(),
                            volume.getNumberOfYCells(),
                            volume.getNumberOfZCells()};

    HDF5Resource filespace(H5Screate_simple(3, dimensions, NULL), H5Sclose);


    hid_t dataset = H5Dcreate(file, name.c_str(), H5T_IEEE_F64LE,
                                   filespace.hid(),
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


    return dataset;
}

///
/// \brief createDatasetForMemroy creates a dataset for the given memory
/// \param volume the volume to read from
/// \param index the index of the memory area to read from
/// \param name the name of the memory (variable name)
/// \param file the file to write to
///
void HDF5Writer::writeMemoryToDataset(const volume::Volume& volume, size_t index, const std::string& name,
                 hid_t dataset)
{
    // Now we will write the portion of data that our process is responsible
    // for (if we are not running MPI, this will default to the whole data)
    // See https://www.hdfgroup.org/HDF5/Tutor/phypecont.html for details on how
    // this works. And also check the C++ example
    // https://www.hdfgroup.org/ftp/HDF5/current/src/unpacked/c++/examples/h5tutr_subset.cpp


    // The number of elements we will write in each direction
    hsize_t count[] = {volume.getNumberOfXCells(),
                       volume.getNumberOfYCells(),
                       volume.getNumberOfZCells()};

    // The offset (where we will start writing data
    hsize_t offset[] = {0, 0, 0};

    HDF5Resource filespace(H5Dget_space(dataset), H5Sclose);

    // Last two arguments are NULL since we accept the default value
    HDF5_SAFE_CALL(H5Sselect_hyperslab(filespace.hid(), H5S_SELECT_SET, offset, NULL, count,
            NULL));

    // We need a temporary memory space to hold the data
    HDF5Resource memspace(H5Screate_simple(3, count, NULL), H5Sclose);


    // Here we only support double writing at the moment
    //static_assert(std::is_same<real, double>::value, "HDF5 only supports double for now");




    std::vector<real> dataTmp(volume.getNumberOfXCells() * volume.getNumberOfYCells() * volume.getNumberOfZCells());
    volume.copyInternalCells(index, dataTmp.data(), dataTmp.size());
    std::vector<double> data(volume.getNumberOfXCells() * volume.getNumberOfYCells() * volume.getNumberOfZCells());
    std::copy(dataTmp.begin(), dataTmp.end(), data.begin());
    // Then we write the data as we normally would.
    HDF5_SAFE_CALL(H5Dwrite(dataset, H5T_NATIVE_DOUBLE,
                memspace.hid(), filespace.hid(), H5P_DEFAULT,
                data.data()));

    writeString(dataset, "vsType", "variable");
    writeString(dataset, "vsMesh", "grid");
    writeString(dataset, "vsCentering", "zonal");
}


void HDF5Writer::writeMemory(const volume::Volume& volume, size_t index,
                             const std::string& name,
                             hid_t file)
{



    hid_t dataset = createDatasetForMemory(volume, index, name, file);
    writeMemoryToDataset(volume, index, name, dataset);
    HDF5_SAFE_CALL(H5Dclose(dataset));


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

    HDF5_SAFE_CALL(H5Awrite(attribute.hid(), H5T_NATIVE_FLOAT, values.data()));
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

