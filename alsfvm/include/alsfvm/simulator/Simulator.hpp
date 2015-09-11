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

///
    class Simulator {
    public:
        Simulator(const SimulatorParameters& simulatorParameters,
                  std::shared_ptr<grid::Grid> & grid,
                  volume::VolumeFactory& volumeFactory,
                  integrator::IntegratorFactory& integratorFactory,
                  boundary::BoundaryFactory& boundaryFactory,
                  numflux::NumericalFluxFactory& numericalFluxFactory,
                  equation::CellComputerFactory& cellComputerFactory,
                  std::shared_ptr<memory::MemoryFactory>& memoryFactory,
                  std::shared_ptr<init::InitialData>& initialData,
                  real endTime);



        bool atEnd();
        void performStep();
        void callWriters();
        void addWriter(std::shared_ptr<io::Writer>& writer);
        real getCurrentTime() const;

        real getEndTime() const;

    private:

        real computeTimestep();
        void checkConstraints();
        void incrementSolution();

        TimestepInformation timestepInformation;
        std::shared_ptr<grid::Grid> grid;
        std::shared_ptr<numflux::NumericalFlux> numericalFlux;
        std::shared_ptr<integrator::Integrator> integrator;
        std::shared_ptr<boundary::Boundary> boundary;
        std::vector<std::shared_ptr<volume::Volume> > conservedVolumes;
        std::shared_ptr<volume::Volume> extraVolume;
        std::shared_ptr<equation::CellComputer> cellComputer;


        std::vector<std::shared_ptr<io::Writer> > writers;
        std::shared_ptr<init::InitialData> initialData;

        const real cflNumber;
        const real endTime;
    };
} // namespace alsfvm
} // namespace simulator
