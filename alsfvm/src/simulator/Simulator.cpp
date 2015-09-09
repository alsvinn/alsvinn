#include "alsfvm/simulator/Simulator.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm { namespace simulator {

Simulator::Simulator(const SimulatorParameters& simulatorParameters,
                     std::shared_ptr<grid::Grid> & grid,
                     volume::VolumeFactory &volumeFactory,
                     integrator::IntegratorFactory &integratorFactory,
                     boundary::BoundaryFactory &boundaryFactory,
                     numflux::NumericalFluxFactory &numericalFluxFactory,
                     equation::CellComputerFactory &cellComputerFactory,
                     std::shared_ptr<memory::MemoryFactory>& memoryFactory)
    : cflNumber(simulatorParameters.getCFLNumber()),
      grid(grid),
      numericalFlux(numericalFluxFactory.createNumericalFlux(*grid)),
      integrator(integratorFactory.createIntegrator(numericalFlux)),
      boundary(boundaryFactory.createBoundary(numericalFlux->getNumberOfGhostCells())),
      cellComputer(cellComputerFactory.createComputer())
{
    const size_t nx = grid->getDimensions().x;
    const size_t ny = grid->getDimensions().y;
    const size_t nz = grid->getDimensions().z;

    for (size_t i = 0; i < integrator->getNumberOfSubsteps() + 1; ++i) {
        conservedVolumes.push_back(
                volumeFactory.createConservedVolume(nx, ny, nz,
                                                      numericalFlux->getNumberOfGhostCells()));
    }

    extraVolume = volumeFactory.createExtraVolume(nx, ny, nz,
                                                  numericalFlux->getNumberOfGhostCells());
}

void Simulator::performStep()
{
    incrementSolution();
    checkConstraints();
    callWriters();
}

void Simulator::addWriter(std::shared_ptr<io::Writer> &writer)
{
    writers.push_back(writer);
}

void Simulator::callWriters()
{
    for(auto writer : writers) {
        writer->write(*conservedVolumes[0],
                      *extraVolume,
                      *grid,
                      timestepInformation);
    }
}

real Simulator::computeTimestep()
{
    const size_t dimension = grid->getActiveDimension();
    real waveSpeedTotal = 0;

    for (size_t direction = 0; direction < dimension; ++direction) {
        const real waveSpeed =
                cellComputer->computeMaxWaveSpeed(*conservedVolumes[0], *extraVolume, direction);
        const real cellLength = grid->getCellLengths()[direction];
        waveSpeedTotal += waveSpeed / cellLength;
    }



    const real dt = cflNumber / waveSpeedTotal;

    return dt;
}

void Simulator::checkConstraints()
{
    const bool obeys = cellComputer->obeysConstraints(*conservedVolumes[0],
            *extraVolume);
    if (!obeys) {
        THROW("Simulation state does not obey constraints! "
              << " At time " << timestepInformation.getCurrentTime()
              << ", number of timesteps performed " << timestepInformation.getNumberOfStepsPerformed());
    }
}

void Simulator::incrementSolution()
{
    const real dt = computeTimestep();
    for (size_t substep = 0; substep < integrator->getNumberOfSubsteps(); ++substep) {
        auto& conservedNext = conservedVolumes[substep + 1];
        integrator->performSubstep(conservedVolumes, grid->getCellLengths(), dt, *conservedNext, 0);
    }
    conservedVolumes[0].swap(conservedVolumes.back());

    cellComputer->computeExtraVariables(*conservedVolumes[0], *extraVolume);
}

}
}
