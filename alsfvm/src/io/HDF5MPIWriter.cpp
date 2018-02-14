#include "alsfvm/io/HDF5MPIWriter.hpp"
#include <mutex>

#include "alsfvm/io/hdf5_utils.hpp"
#include "alsfvm/io/io_utils.hpp"
#include <boost/filesystem.hpp>
#include "alsutils/log.hpp"
#include <H5FDmpi.h>
#include <H5FDmpio.h>

// It seems like HDF5 doesn't like to be accessed from different threads...
static std::mutex mutex;
namespace alsfvm {
namespace io {

HDF5MPIWriter::HDF5MPIWriter(const std::string& basefileName,
    const std::vector<std::string>& groupNames,
    size_t groupIndex,
    bool newFile,
    MPI_Comm mpiCommunicator,
    MPI_Info mpiInfo)
    : HDF5Writer(basefileName),
      groupNames(groupNames),
      groupIndex(groupIndex),
      newFile(newFile),
      mpiCommunicator(mpiCommunicator),
      mpiInfo(mpiInfo) {

}

void HDF5MPIWriter::write(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {

    // for hdf5, often the version we use is not thread safe.
    std::unique_lock<std::mutex> lock(mutex);
    std::string name = getOutputname(basefileName, snapshotNumber);
    std::string h5name = name + std::string(".h5");

    HDF5Resource plist(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);

    HDF5_SAFE_CALL(H5Pset_fapl_mpio(plist.hid(), mpiCommunicator, mpiInfo));
    std::unique_ptr<HDF5Resource> file;

    if (!newFile) {
        //HDF5_SAFE_CALL(H5Pset_fapl_stdio (plist.hid()));
        ALSVINN_LOG(INFO, "Opening file " << h5name);

        HDF5_MAKE_RESOURCE(file, H5Fopen(h5name.c_str(),
                // H5F_ACC_TRUNC,
                H5F_ACC_RDWR,
                plist.hid()), H5Fclose);


    } else {
        ALSVINN_LOG(INFO, "Creating new file " << h5name)
        HDF5_MAKE_RESOURCE(file, H5Fcreate(h5name.c_str(),
                H5F_ACC_TRUNC,
                H5P_DEFAULT,
                plist.hid()), H5Fclose);

    }

    if (newFile) {
        ALSVINN_LOG(INFO, "Writing grid to file " << h5name);
        writeGrid(file->hid(), grid);
    }

    HDF5Resource accessList(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
    HDF5_SAFE_CALL(H5Pset_dxpl_mpio(accessList.hid(), H5FD_MPIO_INDEPENDENT));

    writeVolume(conservedVariables, file->hid(), accessList.hid());
    writeVolume(extraVariables, file->hid(), accessList.hid());
    snapshotNumber++;

}

std::unique_ptr<HDF5Resource> HDF5MPIWriter::createDatasetForMemory(
    const volume::Volume& volume, size_t index, const std::string& name,
    hid_t file) {
    std::unique_ptr<HDF5Resource> dataset = NULL;

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
                volume.getNumberOfZCells()
            };
        //auto groupExistsStatus = H5Gget_objinfo (file, groupName.c_str(), 0, NULL);

        // see https://support.hdfgroup.org/HDF5/doc/RM/RM_H5L.html#Link-Exists
        auto groupExistsStatus = H5Lexists(file, groupName.c_str(), H5P_DEFAULT);


        // Do clever pre checking:
        if (groupExistsStatus && groupName != groupNames[groupIndex]) {
            auto datasetExistsStatusCheck = H5Lexists(file,
                    (groupName + "/" + name).c_str(), H5P_DEFAULT);

            if (datasetExistsStatusCheck > 0) {
                continue;
            }
        }

        // check if the group exists or not
        std::unique_ptr<HDF5Resource> group;

        if (groupExistsStatus > 0) {
            ALSVINN_LOG(INFO, "Opening hdf5 group " << groupName);
            HDF5_MAKE_RESOURCE(group, H5Gopen(file, groupName.c_str(), H5P_DEFAULT),
                H5Gclose);
        } else if (groupExistsStatus == 0) {
            ALSVINN_LOG(INFO, "Creating HDF5 group  " << groupName);
            HDF5_MAKE_RESOURCE(group, H5Gcreate(file, groupName.c_str(),
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT), H5Gclose);

        } else {
            THROW("HDF5 failure in checking if group name " << groupName << " exists.");
        }


        auto datasetExistsStatus = H5Lexists(file, (groupName + "/" + name).c_str(),
                H5P_DEFAULT);
        std::unique_ptr<HDF5Resource> dataset_tmp;

        if (datasetExistsStatus > 0) {
            ALSVINN_LOG(INFO, "Opening dataset " << groupName << "/" << name);
            HDF5_MAKE_RESOURCE(dataset_tmp, H5Dopen(group->hid(), name.c_str(),
                    H5P_DEFAULT), H5Dclose);
        } else if (datasetExistsStatus == 0) {
            ALSVINN_LOG(INFO, "Creating dataset " << groupName << "/" << name);
            std::unique_ptr<HDF5Resource> filespace;
            HDF5_MAKE_RESOURCE(filespace, H5Screate_simple(3, dimensions, NULL), H5Sclose);

            HDF5_MAKE_RESOURCE(dataset_tmp, H5Dcreate(group->hid(), name.c_str(),
                    H5T_IEEE_F64LE,
                    filespace->hid(),
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);


        } else {
            THROW("HDF5 failure in checking if dataset name " << name <<
                " exists. (group = " << groupName << ").");
        }


        if (groupName == groupNames[groupIndex]) {
            dataset = std::move(dataset_tmp);
        }

    }

    if (!dataset) {
        THROW("Dataset not found or created " << groupNames[groupIndex] << "/" << name);
    }

    return dataset;
}

}
}
