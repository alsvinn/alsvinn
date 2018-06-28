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

#pragma once
#include "alsfvm/io/WriterFactory.hpp"
#include <mpi.h>
namespace alsuq {
namespace io {

class MPIWriterFactory : public alsfvm::io::WriterFactory {
public:

    MPIWriterFactory(const std::vector<std::string>& groupNames,
        size_t groupIndex,
        bool createFile,
        MPI_Comm mpiCommunicator,
        MPI_Info mpiInfo);


    alsfvm::shared_ptr<alsfvm::io::Writer>
    createWriter(const std::string& name, const std::string& baseFilename,
        const alsfvm::io::Parameters& parameters) override;

private:
    std::vector<std::string> groupNames;
    size_t groupIndex;
    bool createFile;
    MPI_Comm mpiCommunicator;
    MPI_Info mpiInfo;
};
} // namespace io
} // namespace alsuq
