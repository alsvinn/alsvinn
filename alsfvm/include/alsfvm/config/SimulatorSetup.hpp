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
#include <iostream>
#include "alsfvm/simulator/Simulator.hpp"
#include <memory>
#include <boost/property_tree/ptree.hpp>
#include "alsfvm/diffusion/DiffusionOperator.hpp"
#include "alsfvm/init/Parameters.hpp"
#include "alsfvm/io/WriterFactory.hpp"
#ifdef ALSVINN_USE_MPI
    #include <mpi.h>
    #include "alsfvm/mpi/domain/DomainInformation.hpp"
#endif

namespace alsfvm {
namespace config {

class SimulatorSetup {
public:
    typedef boost::property_tree::ptree ptree;
    std::pair<alsfvm::shared_ptr<simulator::Simulator>,
        alsfvm::shared_ptr<init::InitialData> >
        readSetupFromFile(const std::string& filename);


    void setWriterFactory(std::shared_ptr<io::WriterFactory> writerFactory);

#ifdef ALSVINN_USE_MPI

    //! Call to enable mpi. Has to be called *before* readSetupFromFile.
    void enableMPI(MPI_Comm communicator, int multiX, int multiY, int multiZ);

    //! Call to enable mpi. Has to be called *before* readSetupFromFile.
    void enableMPI(mpi::ConfigurationPtr configuration, int multiX, int multiY,
        int multiZ);
#endif
protected:

    alsfvm::shared_ptr<init::InitialData> createInitialData(
        const ptree& configuration);
    alsfvm::shared_ptr<grid::Grid> createGrid(const ptree& configuration);
    real readEndTime(const ptree& configuration);
    std::string readEquation(const ptree& configuration);
    std::string readReconstruciton(const ptree& configuration);
    real readCFLNumber(const ptree& configuration);
    std::string readIntegrator(const ptree& configuration);

    alsfvm::shared_ptr<io::Writer> createWriter(const ptree& configuration);
    std::string readPlatform(const ptree& configuration);
    std::string readBoundary(const ptree& configuration);
    init::Parameters readParameters(const ptree& configuration);
    alsfvm::shared_ptr<diffusion::DiffusionOperator> createDiffusion(
        const ptree& configuration,
        const grid::Grid& grid,
        const simulator::SimulatorParameters& simulatorParameters,
        alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration,
        alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
        volume::VolumeFactory& volumeFactory);

    std::string readName(const ptree& configuration);
    std::vector<io::WriterPointer> createFunctionals(const ptree& configuration,
        volume::VolumeFactory& volumeFactory);
    void readEquationParameters(const ptree& configuration,
        simulator::SimulatorParameters& parameters);

    std::string readFlux(const ptree& configuration);

    std::shared_ptr<io::WriterFactory> writerFactory{new io::WriterFactory};
    std::string basePath;


#ifdef ALSVINN_USE_MPI
    mpi::domain::DomainInformationPtr decomposeGrid(const
        alsfvm::shared_ptr<grid::Grid>& grid);
    bool useMPI{false};
    mpi::ConfigurationPtr mpiConfiguration;
    int multiX;
    int multiY;
    int multiZ;
#endif
};
} // namespace alsfvm
} // namespace config
