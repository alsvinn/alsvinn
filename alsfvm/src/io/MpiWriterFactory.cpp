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

#include "alsfvm/io/MpiWriterFactory.hpp"
#include "alsfvm/io/NetCDFMPIWriter.hpp"
#include "alsfvm/io/PythonScript.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsfvm {
namespace io {

MpiWriterFactory::MpiWriterFactory(mpi::ConfigurationPtr configuration)
    : configuration(configuration) {

}

alsfvm::shared_ptr<Writer> MpiWriterFactory::createWriter(
    const std::string& name,
    const std::string& baseFilename, const io::Parameters& parameters) {
    alsfvm::shared_ptr<Writer> writer;

    auto parameterCopy = parameters;
    parameterCopy.addIntegerParameter("mpi_rank", configuration->getRank());
    parameterCopy.addIntegerParameter("mpi_size",
        configuration->getNumberOfProcesses());

    if (name == "netcdf") {
        writer.reset(new NetCDFMPIWriter(baseFilename, {""}, 0, true,
                configuration->getCommunicator(),
                configuration->getInfo()));
    } else if (name == "python") {
        writer.reset(new PythonScript(baseFilename, parameters, configuration));
    } else {
        THROW("Unknown writer " << name << std::endl << "Does not have MPI support.");
    }

    return writer;
}

}
}
