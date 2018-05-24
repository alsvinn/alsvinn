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

#include "alsutils/mpi/Configuration.hpp"
#include "alsutils/mpi/safe_call.hpp"

namespace alsutils {
namespace mpi {

Configuration::Configuration(MPI_Comm communicator,
    const std::string& platform)
    : communicator(communicator), platform(platform) {
    MPI_Comm_rank(communicator, &nodeNumber);
    MPI_Comm_size(communicator, &numberOfNodes);
    info = MPI_INFO_NULL;
}

MPI_Comm Configuration::getCommunicator() {
    return communicator;
}

int Configuration::getRank() const {
    return nodeNumber;
}

int Configuration::getNumberOfProcesses() const {
    return numberOfNodes;
}

MPI_Info Configuration::getInfo() {
    return info;
}

std::string Configuration::getPlatform() const {
    return platform;
}

alsfvm::shared_ptr<Configuration> Configuration::makeSubConfiguration(int color,
    int newRank) const {
    MPI_Comm newCommunicator;
    MPI_SAFE_CALL(MPI_Comm_split(communicator, color, newRank, &newCommunicator));

    return ConfigurationPtr(new Configuration(newCommunicator, platform));
}

}
}
