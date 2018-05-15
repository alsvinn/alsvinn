#include "alsfvm/io/NetCDFWriter.hpp"
#include "alsfvm/io/netcdf_utils.hpp"
#include "alsfvm/io/io_utils.hpp"
#include "alsutils/log.hpp"
#include "alsfvm/io/netcdf_write_report.hpp"
namespace alsfvm {
namespace io {

NetCDFWriter::NetCDFWriter(const std::string& basefileName)
    : basefileName(basefileName) {

}

void NetCDFWriter::write(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    netcdf_raw_ptr file;
    auto filename = getFilename();
    NETCDF_SAFE_CALl(nc_create(filename.c_str(),  NC_CLOBBER | NC_NETCDF4, &file));
    netcdfWriteReport(file);
    writeToFile(file, conservedVariables, extraVariables, grid,
        timestepInformation);
    NETCDF_SAFE_CALl(nc_close(file));
}

void NetCDFWriter::writeToFile(netcdf_raw_ptr file,
    const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    auto dimensions = createDimensions(file, conservedVariables);
    writeVolume(file, conservedVariables, dimensions);
    writeVolume(file, extraVariables, dimensions);

}

std::array<netcdf_raw_ptr, 3> NetCDFWriter::createDimensions(
    netcdf_raw_ptr baseGroup, const volume::Volume& volume) {

    std::array<netcdf_raw_ptr, 3> dimensions;
    netcdf_raw_ptr xdim, ydim, zdim;


    NETCDF_SAFE_CALl(nc_def_dim(baseGroup, "x", volume.getNumberOfXCells(),
            &xdim));
    NETCDF_SAFE_CALl(nc_def_dim(baseGroup, "y", volume.getNumberOfYCells(),
            &ydim));
    NETCDF_SAFE_CALl(nc_def_dim(baseGroup, "z", volume.getNumberOfZCells(),
            &zdim));

    dimensions[0] = xdim;
    dimensions[1] = ydim;
    dimensions[2] = zdim;

    return dimensions;

}

void NetCDFWriter::writeMemory(netcdf_raw_ptr baseGroup, netcdf_raw_ptr dataset,
    const volume::Volume& volume, size_t memoryIndex) {
    std::vector<real> dataTmp(volume.getNumberOfXCells() *
        volume.getNumberOfYCells() * volume.getNumberOfZCells());

    volume.copyInternalCells(memoryIndex, dataTmp.data(), dataTmp.size());

    std::vector<double> data(dataTmp.size());
    std::copy(dataTmp.begin(), dataTmp.end(), data.begin());

    NETCDF_SAFE_CALl(nc_put_var_double(baseGroup, dataset, data.data()));
}



void NetCDFWriter::writeVolume(netcdf_raw_ptr baseGroup,
    const volume::Volume& volume, std::array<netcdf_raw_ptr, 3> dimensions) {

    for (size_t variable = 0; variable < volume.getNumberOfVariables();
        ++variable) {
        auto dataset = makeDataset(baseGroup, volume, variable, dimensions);


        writeMemory(dataset.first, dataset.second, volume, variable);
    }
}

std::pair<netcdf_raw_ptr, netcdf_raw_ptr> NetCDFWriter::makeDataset(
    netcdf_raw_ptr baseGroup, const volume::Volume& volume, size_t memoryIndex,
    std::array<netcdf_raw_ptr, 3> dimensions) {

    netcdf_raw_ptr datasetId;
    NETCDF_SAFE_CALl(nc_def_var(baseGroup, volume.getName(memoryIndex).c_str(),
            NC_DOUBLE, 3, dimensions.data(), &datasetId));
    return std::make_pair(baseGroup, datasetId);

}

std::string NetCDFWriter::getFilename() {

    std::string name = getOutputname(basefileName, snapshotNumber);
    std::string h5name = name + std::string(".nc");
    snapshotNumber++;
    return h5name;

}



}
}
