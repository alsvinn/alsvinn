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
#include "alsfvm/simulator/AbstractSimulator.hpp"
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/integrator/IntegratorFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/simulator/TimestepInformation.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/init/InitialData.hpp"
#include "alsfvm/simulator/ConservedSystem.hpp"
#include "alsfvm/diffusion/DiffusionOperator.hpp"
#include <vector>
#include <memory>

#ifdef ALSVINN_USE_MPI
    #include "alsfvm/mpi/CellExchanger.hpp"
#endif

namespace alsfvm {
namespace simulator {

///
/// \brief The Simulator class contains all the neccesary tools for running the
/// whole simulation.
///
/// How to use:
/// \code{.cpp}
/// // save first timestep
/// simulator.callWriters();
/// while (!simulator.atEnd()) {
///    simulator.performStep();
/// }
/// \endcode

///
class Simulator : public AbstractSimulator {
public:
    ///
    /// \brief Simulator
    /// \param simulatorParameters
    /// \param grid
    /// \param volumeFactory
    /// \param integratorFactory
    /// \param boundaryFactory
    /// \param numericalFluxFactory
    /// \param cellComputerFactory
    /// \param memoryFactory
    /// \param endTime
    /// \param deviceConfiguration
    /// \param equationName
    /// \param diffusionOperator the diffusion operator to use
    /// \param name the name of the simulator
    ///
    Simulator(const SimulatorParameters& simulatorParameters,
        alsfvm::shared_ptr<grid::Grid>& grid,
        volume::VolumeFactory& volumeFactory,
        integrator::IntegratorFactory& integratorFactory,
        boundary::BoundaryFactory& boundaryFactory,
        numflux::NumericalFluxFactory& numericalFluxFactory,
        equation::CellComputerFactory& cellComputerFactory,
        alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
        real endTime,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration,
        std::string& equationName,
        alsfvm::shared_ptr<alsfvm::diffusion::DiffusionOperator> diffusionOperator,
        const std::string& name
    );

    ~Simulator();



    ///
    /// \return true if the simulation is finished, false otherwise.
    ///
    bool atEnd() override;

    ///
    /// Performs one timestep
    ///
    void performStep() override;

    ///
    /// Calls the writers.
    ///
    void callWriters() override;

    ///
    /// \brief addWriter adds a writer, this will be called every time callWriter is called
    /// \param writer
    ///
    void addWriter(alsfvm::shared_ptr<io::Writer> writer) override;

    //! Adds a timestep adjuster.
    //!
    //! The timestep adjuster is run as
    //! \code{.cpp}
    //! real newTimestep = someInitialValueFromCFL;
    //! for (auto adjuster : timestepAdjusters) {
    //!      newTimestep = adjuster(newTimestep);
    //! }
    //! \endcode
    //!
    //! the timestep adjuster is used to save at specific times.
    void addTimestepAdjuster(alsfvm::shared_ptr<integrator::TimestepAdjuster>&
        adjuster) override;

    ///
    /// \return the current simulation time.
    ///
    real getCurrentTime() const override;

    ///
    /// \return the end time of the simulation.
    ///
    real getEndTime() const override;

    ///
    /// Updates the simulation state.
    ///
    /// \param conservedVolume the conservedVolume to update to
    ///
    /// \note This does not need to be on the same size as the conserved volume,
    ///       interpolation will be done.
    ///
    void setSimulationState(const volume::Volume& conservedVolume);

    std::string getPlatformName() const;

    std::string getEquationName() const;

    void setInitialValue(alsfvm::shared_ptr<init::InitialData>& initialData);

    const std::shared_ptr<grid::Grid>& getGrid() const override;
    std::shared_ptr<grid::Grid>& getGrid() override;

    void finalize() override;

#ifdef ALSVINN_USE_MPI
    void setCellExchanger(mpi::CellExchangerPtr value);
#endif

    std::string getName() const;

private:


    void checkConstraints();
    void incrementSolution();
    void doCellExchange(volume::Volume& volume);

    SimulatorParameters simulatorParameters;
    volume::VolumeFactory volumeFactory;
    TimestepInformation timestepInformation;
    alsfvm::shared_ptr<grid::Grid> grid;
    alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux;
    alsfvm::shared_ptr<integrator::System> system;
    alsfvm::shared_ptr<integrator::Integrator> integrator;
    alsfvm::shared_ptr<boundary::Boundary> boundary;
    std::vector<alsfvm::shared_ptr<volume::Volume> > conservedVolumes;
    alsfvm::shared_ptr<volume::Volume> extraVolume;
    alsfvm::shared_ptr<equation::CellComputer> cellComputer;

    alsfvm::shared_ptr<alsfvm::diffusion::DiffusionOperator> diffusionOperator;
    std::vector<alsfvm::shared_ptr<io::Writer> > writers;
    alsfvm::shared_ptr<init::InitialData> initialData;

    const real cflNumber;
    const real endTime;
    const std::string equationName;
    const std::string platformName;
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;

#ifdef ALSVINN_USE_MPI
    mpi::CellExchangerPtr cellExchanger;
#endif

    const std::string name;
};
} // namespace alsfvm
} // namespace simulator
