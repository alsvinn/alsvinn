/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsuq/io/MPIWriterFactory.hpp"
#include "alsfvm/io/HDF5MPIWriter.hpp"
#include "alsfvm/io/NetCDFMPIWriter.hpp"
namespace alsuq {
namespace io {

MPIWriterFactory::MPIWriterFactory(const std::vector<std::string>& groupNames,
    size_t groupIndex,
    bool createFile,
    MPI_Comm mpiCommunicator, MPI_Info mpiInfo)

    : groupNames(groupNames), groupIndex(groupIndex), createFile(createFile),
      mpiCommunicator(mpiCommunicator),
      mpiInfo(mpiInfo) {

}

alsfvm::shared_ptr<alsfvm::io::Writer> MPIWriterFactory::createWriter(
    const std::string& name,
    const std::string& baseFilename) {
    alsfvm::shared_ptr<alsfvm::io::Writer> writer;

    if (name == "hdf5") {
        writer.reset(new alsfvm::io::HDF5MPIWriter(baseFilename, groupNames,
                groupIndex, createFile, mpiCommunicator,
                mpiInfo));
    } else if (name == "netcdf") {
        writer.reset(new alsfvm::io::NetCDFMPIWriter(baseFilename, groupNames,
                groupIndex, createFile, mpiCommunicator,
                mpiInfo));

    } else {
        THROW("Unknown writer " << name);
    }

    return writer;
}

}
}
