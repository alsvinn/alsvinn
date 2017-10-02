#include "alsfvm/io/NetCDFMPIWriter.hpp"
#include <pnetcdf.h>
#include "alsutils/log.hpp"
#include "alsutils/mpi/to_mpi_offset.hpp"

namespace alsfvm { namespace io {

NetCDFMPIWriter::NetCDFMPIWriter(const std::string &basefileName,
                                 const std::vector<std::string> &groupNames,
                                 size_t groupIndex, bool newFile,
                                 MPI_Comm mpiCommunicator, MPI_Info mpiInfo)
    : NetCDFWriter(basefileName),
      groupNames(groupNames),
      groupIndex(groupIndex),
      newFile(newFile),
      mpiCommunicator(mpiCommunicator),
      mpiInfo(mpiInfo)
{

}

void NetCDFMPIWriter::write(const volume::Volume &conservedVariables,
                            const volume::Volume &extraVariables,
                            const grid::Grid &grid,
                            const simulator::TimestepInformation &timestepInformation)
{
    netcdf_raw_ptr file;
    auto filename = getFilename();
    netcdf_raw_ptr timeVar;
    if (newFile) {

      NETCDF_SAFE_CALl(ncmpi_create(mpiCommunicator, filename.c_str(),  NC_64BIT_DATA|NC_CLOBBER,
                                   mpiInfo, &file));


        // write current time
        netcdf_raw_ptr timeDim;
        NETCDF_SAFE_CALl(ncmpi_def_dim(file, "t", 1, &timeDim));


        NETCDF_SAFE_CALl(ncmpi_def_var(file, "time", NC_DOUBLE, 1, &timeDim,
                                       &timeVar));



    }
    else {
      NETCDF_SAFE_CALl(ncmpi_open(mpiCommunicator, filename.c_str(), NC_WRITE| NC_64BIT_DATA,
                            mpiInfo, &file));
      NETCDF_SAFE_CALl(ncmpi_redef(file));
    }
    writeToFile(file, conservedVariables, extraVariables,
                grid, timestepInformation, newFile);

    if (newFile) {

        double currentTime = timestepInformation.getCurrentTime();
        NETCDF_SAFE_CALl(ncmpi_put_var_double_all(file, timeVar, &currentTime));
    }

    NETCDF_SAFE_CALl(ncmpi_close(file));
}

NetCDFMPIWriter::dimension_vector NetCDFMPIWriter::createDimensions(netcdf_raw_ptr baseGroup, const grid::Grid &grid, bool newFile)
{
    std::array<netcdf_raw_ptr,3> dimensions;
    netcdf_raw_ptr xdim, ydim, zdim;

    if (newFile) {
        NETCDF_SAFE_CALl(ncmpi_def_dim(baseGroup, "x", grid.getGlobalSize()[0],
                                       &xdim));
        NETCDF_SAFE_CALl(ncmpi_def_dim(baseGroup, "y", grid.getGlobalSize()[1],
                                       &ydim));
        NETCDF_SAFE_CALl(ncmpi_def_dim(baseGroup, "z", grid.getGlobalSize()[2],
                                       &zdim));
    } else {
        NETCDF_SAFE_CALl(ncmpi_inq_dimid(baseGroup, "x", &xdim));
        NETCDF_SAFE_CALl(ncmpi_inq_dimid(baseGroup, "y", &ydim));
        NETCDF_SAFE_CALl(ncmpi_inq_dimid(baseGroup, "z", &zdim));
    }

    dimensions[0] = xdim;
    dimensions[1] = ydim;
    dimensions[2] = zdim;

    return dimensions;
}

std::vector<netcdf_raw_ptr>
NetCDFMPIWriter::makeDataset(netcdf_raw_ptr baseGroup,
                             const volume::Volume &volume,
                             std::array<netcdf_raw_ptr, 3> dimensions)
{
    std::vector<netcdf_raw_ptr> datasets;


    for (const auto& groupName : groupNames) {

        for (size_t memoryIndex = 0; memoryIndex < volume.getNumberOfVariables();
             ++memoryIndex) {
            netcdf_raw_ptr dataset;

            std::string groupnamePrefix = "";

            if (groupName.size() > 0) {
                groupnamePrefix = groupName + "_";
            }
            auto memoryName = groupnamePrefix + volume.getName(memoryIndex) ;


            NETCDF_SAFE_CALl(ncmpi_def_var(baseGroup, memoryName.c_str(), NC_DOUBLE, 3,
                                           dimensions.data(), &dataset));

            if (groupName == groupNames[groupIndex]) {
                datasets.push_back(dataset);
            }
        }

    }

    return datasets;
}

void NetCDFMPIWriter::writeToFile(netcdf_raw_ptr file,
                                  const volume::Volume &conservedVariables,
                                  const volume::Volume &extraVariables,
                                  const grid::Grid &grid,
                                  const simulator::TimestepInformation &timestepInformation,
                                  bool newFile)
{


    auto dimensions = createDimensions(file, grid, newFile);


    auto datasetsConserved = makeDataset(file, conservedVariables, dimensions);
    auto datasetsExtra = makeDataset(file, extraVariables, dimensions);

    NETCDF_SAFE_CALl(ncmpi_enddef(file));
    writeVolume(file, conservedVariables, dimensions, datasetsConserved, grid);
    writeVolume(file, extraVariables, dimensions, datasetsExtra, grid);

}

void NetCDFMPIWriter::writeMemory(netcdf_raw_ptr baseGroup,
                                  netcdf_raw_ptr dataset,
                                  const volume::Volume &volume,
                                  size_t memoryIndex,
                                  const grid::Grid& grid)
{
    std::vector<real> dataTmp(volume.getNumberOfXCells() * volume.getNumberOfYCells() * volume.getNumberOfZCells());

    volume.copyInternalCells(memoryIndex, dataTmp.data(), dataTmp.size());

    std::vector<double> data(dataTmp.size());
    std::copy(dataTmp.begin(), dataTmp.end(), data.begin());
    auto globalPosition = alsutils::mpi::to_mpi_offset(grid.getGlobalPosition());
    auto globalSize = alsutils::mpi::to_mpi_offset(grid.getDimensions());
    NETCDF_SAFE_CALl(ncmpi_put_vara_double_all(baseGroup, dataset, globalPosition.data(),
                                               globalSize.data(),
                                               data.data()));
}

void NetCDFMPIWriter::writeVolume(netcdf_raw_ptr baseGroup, const volume::Volume &volume, std::array<netcdf_raw_ptr, 3> dimensions,
                                  const std::vector<netcdf_raw_ptr>& datasets,
                                  const grid::Grid& grid)
{
    for (size_t memoryIndex = 0; memoryIndex < volume.getNumberOfVariables();
                 ++memoryIndex) {
        auto dataset = datasets[memoryIndex];

        writeMemory(baseGroup, dataset, volume, memoryIndex, grid);
    }
}


}
}
