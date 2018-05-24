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
#include "alsutils/types.hpp"
#include <mpi.h>
namespace alsutils {
namespace mpi {

class Configuration {
public:
    Configuration(MPI_Comm communicator,
        const std::string& platform = "cpu");

    MPI_Comm getCommunicator();

    int getRank() const;

    int getNumberOfProcesses() const;

    MPI_Info getInfo();
    std::string getPlatform() const;

    //! Essentially maps to MPI_Comm_split. See tutorial here:
    //! http://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
    alsfvm::shared_ptr<Configuration> makeSubConfiguration(int color,
        int newRank) const;

private:
    MPI_Comm communicator;

    int nodeNumber;
    int numberOfNodes;
    MPI_Info info;
    const std::string platform = "cpu";
};

typedef alsfvm::shared_ptr<Configuration> ConfigurationPtr;
} // namespace mpi
} // namespace alsutils
