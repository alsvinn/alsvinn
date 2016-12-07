#include "alsfvm/io/HDF5MPIWriter.hpp"
#include <mutex>

#include "alsfvm/io/hdf5_utils.hpp"
#include "alsfvm/io/io_utils.hpp"

// It seems like HDF5 doesn't like to be accessed from different threads...
static std::mutex mutex;
namespace alsfvm { namespace io {

HDF5MPIWriter::HDF5MPIWriter(const std::string &basefileName,
                             const std::vector<std::string> &groupNames,
                             size_t groupIndex,
                             MPI_Comm mpiCommunicator,
                             MPI_Info mpiInfo)
    : HDF5Writer(basefileName),
      groupNames(groupNames),
      groupIndex(groupIndex),
      mpiCommunicator(mpiCommunicator),
      mpiInfo(mpiInfo)
{

}

void HDF5MPIWriter::write(const volume::Volume &conservedVariables,
                          const volume::Volume &extraVariables,
                          const grid::Grid &grid,
                          const simulator::TimestepInformation &timestepInformation)
{
    // for hdf5, often the version we use is not thread safe.
    std::unique_lock<std::mutex> lock(mutex);
    std::string name = getOutputname(basefileName, snapshotNumber);
    std::string h5name = name + std::string(".h5");

    HDF5Resource plist(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);

    H5Pset_fapl_mpio(plist.hid(), mpiCommunicator, mpiInfo);

    HDF5Resource file(H5Fcreate(h5name.c_str(),
                                H5F_ACC_TRUNC, H5P_DEFAULT,
                                plist.hid()), H5Fclose);

    writeGrid(file.hid(), grid);
    writeVolume(conservedVariables, file.hid());
    writeVolume(extraVariables, file.hid());
    snapshotNumber++;
}

hid_t HDF5MPIWriter::createDatasetForMemory(const volume::Volume &volume, size_t index, const std::string &name, hid_t file)
{
    hid_t dataset;

    // We loop through each group name, and create
    // group/name dataset (this needs to be done in the same order for all MPI
    // tasks)
    for (auto& groupName : groupNames) {
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
        auto groupExistsStatus = H5Gget_objinfo (file, groupName.c_str(), 0, NULL);

        // check if the group exists or not
        hid_t groupHid;
        if (groupExistsStatus == 0) {
            groupHid = H5Gopen1(file, groupName.c_str());
        } else {
            groupHid = H5Gcreate2(file, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }
        HDF5Resource group(groupHid, H5Gclose);
        HDF5Resource filespace(H5Screate_simple(3, dimensions, NULL), H5Sclose);


        hid_t dataset_tmp = H5Dcreate(group.hid(), name.c_str(), H5T_IEEE_F64LE,
                                       filespace.hid(),
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (groupName == groupNames[groupIndex]) {
            dataset = dataset_tmp;
        } else {
            HDF5_SAFE_CALL(H5Dclose(dataset_tmp));
        }

    }
    return dataset;
}

}
}
