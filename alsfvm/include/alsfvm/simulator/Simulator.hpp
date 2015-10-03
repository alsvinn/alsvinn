#pragma once
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
#include <vector>
#include <memory>

namespace alsfvm { namespace simulator {

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
    class Simulator {
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
        /// \param initialData
        /// \param endTime
        /// \param deviceConfiguration
        /// \param equationName
        ///
        Simulator(const SimulatorParameters& simulatorParameters,
                  alsfvm::shared_ptr<grid::Grid> & grid,
                  volume::VolumeFactory& volumeFactory,
                  integrator::IntegratorFactory& integratorFactory,
                  boundary::BoundaryFactory& boundaryFactory,
                  numflux::NumericalFluxFactory& numericalFluxFactory,
                  equation::CellComputerFactory& cellComputerFactory,
                  alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
                  alsfvm::shared_ptr<init::InitialData>& initialData,
				  real endTime,
				  alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration,
				  std::string& equationName);



        ///
        /// \return true if the simulation is finished, false otherwise.
        ///
        bool atEnd();

        ///
        /// Performs one timestep
        ///
        void performStep();

        ///
        /// Calls the writers.
        ///
        void callWriters();

        ///
        /// \brief addWriter adds a writer, this will be called every time callWriter is called
        /// \param writer
        ///
        void addWriter(alsfvm::shared_ptr<io::Writer>& writer);

        ///
        /// \return the current simulation time.
        ///
        real getCurrentTime() const;

        ///
        /// \return the end time of the simulation.
        ///
        real getEndTime() const;

    private:

        real computeTimestep();
        void checkConstraints();
        void incrementSolution();

        TimestepInformation timestepInformation;
        alsfvm::shared_ptr<grid::Grid> grid;
        alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux;
        alsfvm::shared_ptr<integrator::Integrator> integrator;
        alsfvm::shared_ptr<boundary::Boundary> boundary;
        std::vector<alsfvm::shared_ptr<volume::Volume> > conservedVolumes;
        alsfvm::shared_ptr<volume::Volume> extraVolume;
        alsfvm::shared_ptr<equation::CellComputer> cellComputer;


        std::vector<alsfvm::shared_ptr<io::Writer> > writers;
        alsfvm::shared_ptr<init::InitialData> initialData;

        const real cflNumber;
        const real endTime;
    };
} // namespace alsfvm
} // namespace simulator
