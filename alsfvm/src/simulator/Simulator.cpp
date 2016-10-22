#include "alsfvm/simulator/Simulator.hpp"
#include "alsfvm/error/Exception.hpp"
#include <iostream>

namespace alsfvm { namespace simulator {

Simulator::Simulator(const SimulatorParameters& simulatorParameters,
                     alsfvm::shared_ptr<grid::Grid> & grid,
                     volume::VolumeFactory &volumeFactory,
                     integrator::IntegratorFactory &integratorFactory,
                     boundary::BoundaryFactory &boundaryFactory,
                     numflux::NumericalFluxFactory &numericalFluxFactory,
                     equation::CellComputerFactory &cellComputerFactory,
                     alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
                     alsfvm::shared_ptr<init::InitialData>& initialData,
                     real endTime,
					 alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration,
					 std::string& equationName,
                     alsfvm::shared_ptr<alsfvm::diffusion::DiffusionOperator> diffusionOperator)
    :      grid(grid),
      numericalFlux(numericalFluxFactory.createNumericalFlux(*grid)),
      system(new ConservedSystem(numericalFlux, diffusionOperator)),
      integrator(integratorFactory.createIntegrator(system)),
      boundary(boundaryFactory.createBoundary(system->getNumberOfGhostCells())),
      cellComputer(cellComputerFactory.createComputer()),
      diffusionOperator(diffusionOperator),
      initialData(initialData),
      cflNumber(simulatorParameters.getCFLNumber()),
      endTime(endTime),
      equationName(equationName),
      platformName(deviceConfiguration->getPlatform())
{
    const size_t nx = grid->getDimensions().x;
    const size_t ny = grid->getDimensions().y;
    const size_t nz = grid->getDimensions().z;

    for (size_t i = 0; i < integrator->getNumberOfSubsteps() + 1; ++i) {
        conservedVolumes.push_back(
                volumeFactory.createConservedVolume(nx, ny, nz,
                                                      system->getNumberOfGhostCells()));
    }

    extraVolume = volumeFactory.createExtraVolume(nx, ny, nz,
                                                  system->getNumberOfGhostCells());

   
	// We need to do some extra stuff for the GPU
	// Here we will create the volumes on the CPU, initialize the data on the 
	// cpu, then copy it to the GPU
	if (deviceConfiguration->getPlatform() == "cuda") {
		auto deviceConfigurationCPU = alsfvm::make_shared<DeviceConfiguration>("cpu");
		auto memoryFactoryCPU = alsfvm::make_shared<memory::MemoryFactory>(deviceConfigurationCPU);
		volume::VolumeFactory volumeFactoryCPU(equationName, memoryFactoryCPU);
		auto primitiveVolume = volumeFactoryCPU.createPrimitiveVolume(nx, ny, nz,
			system->getNumberOfGhostCells());
		auto conservedVolumeCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz,
			system->getNumberOfGhostCells());
		auto extraVolumeCPU = volumeFactoryCPU.createExtraVolume(nx, ny, nz,
			system->getNumberOfGhostCells());

        auto simulatorParametersCPU = alsfvm::make_shared<SimulatorParameters>(simulatorParameters);
        simulatorParametersCPU->setPlatform("cpu");
        equation::CellComputerFactory cellComputerFactoryCPU(simulatorParametersCPU, deviceConfigurationCPU);
		auto cellComputerCPU = cellComputerFactoryCPU.createComputer();
		initialData->setInitialData(*conservedVolumeCPU, *extraVolumeCPU, *primitiveVolume, *cellComputerCPU, *grid);
		
		conservedVolumeCPU->copyTo(*conservedVolumes[0]);
		extraVolumeCPU->copyTo(*extraVolume);
		
	}
	else {
		auto primitiveVolume = volumeFactory.createPrimitiveVolume(nx, ny, nz,
			system->getNumberOfGhostCells());
		initialData->setInitialData(*conservedVolumes[0], *extraVolume, *primitiveVolume, *cellComputer, *grid);
	}
    

    boundary->applyBoundaryConditions(*conservedVolumes[0], *grid);
    cellComputer->computeExtraVariables(*conservedVolumes[0], *extraVolume);

}

bool Simulator::atEnd()
{
    return timestepInformation.getCurrentTime() >= endTime;
}

void Simulator::performStep()
{
    incrementSolution();
    checkConstraints();
    callWriters();
}

void Simulator::addWriter(alsfvm::shared_ptr<io::Writer> &writer)
{
    writers.push_back(writer);
}

void Simulator::addTimestepAdjuster(alsfvm::shared_ptr<integrator::TimestepAdjuster> &adjuster)
{
    integrator->addTimestepAdjuster(adjuster);
}

real Simulator::getCurrentTime() const
{
    return timestepInformation.getCurrentTime();
}

real Simulator::getEndTime() const
{
    return endTime;
}

void Simulator::setSimulationState(const volume::Volume &conservedVolume)
{
    conservedVolumes[0]->setVolume(conservedVolume);
    boundary->applyBoundaryConditions(*conservedVolumes[0], *grid);
}

std::string Simulator::getPlatformName() const
{
    return platformName;
}

std::string Simulator::getEquationName() const
{
    return equationName;
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
	real dt = 0;
    for (size_t substep = 0; substep < integrator->getNumberOfSubsteps(); ++substep) {
        auto& conservedNext = conservedVolumes[substep + 1];
        dt = integrator->performSubstep(conservedVolumes, grid->getCellLengths(), dt, cflNumber, *conservedNext, substep, timestepInformation);
        boundary->applyBoundaryConditions(*conservedNext, *grid);
    }
    conservedVolumes[0].swap(conservedVolumes.back());

    cellComputer->computeExtraVariables(*conservedVolumes[0], *extraVolume);
    timestepInformation.incrementTime(dt);
}

}
}
