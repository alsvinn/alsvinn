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

#include "alsfvm/io/WriterFactory.hpp"
#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/io/NetCDFWriter.hpp"
#include "alsfvm/io/PythonScript.hpp"
#include "alsfvm/io/DLLWriter.hpp"

namespace alsfvm {
namespace io {

alsfvm::shared_ptr<Writer> WriterFactory::createWriter(const std::string& name,
    const std::string& baseFilename, const parameters::Parameters& parameters) {


    alsfvm::shared_ptr<Writer> writer;

    if (name == "hdf5") {
        writer.reset(new HDF5Writer(baseFilename));
    } else if (name == "netcdf" ) {
        writer.reset(new NetCDFWriter(baseFilename));
    } else if (name == "python") {
        writer.reset(new PythonScript(baseFilename, parameters));
    } else if (name == "dll") {
        writer.reset(new DLLWriter(baseFilename, parameters));
    } else {
        THROW("Unknown writer type " << name);
    }

    return writer;
}
}
}
