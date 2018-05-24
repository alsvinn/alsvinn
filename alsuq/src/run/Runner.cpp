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

#include "alsuq/run/Runner.hpp"

#include "alsutils/log.hpp"

namespace alsuq {
namespace run {

Runner::Runner(std::shared_ptr<SimulatorCreator> simulatorCreator,
    std::shared_ptr<samples::SampleGenerator> sampleGenerator,
    std::vector<size_t> sampleNumbers,
    alsutils::mpi::ConfigurationPtr mpiConfig,
    const std::string& name)
    : simulatorCreator(simulatorCreator),
      sampleGenerator(sampleGenerator),
      parameterNames(sampleGenerator->getParameterList()),
      sampleNumbers(sampleNumbers),
      mpiConfig(mpiConfig),
      name(name)

{

}

void Runner::run() {
    std::shared_ptr<alsfvm::grid::Grid> grid;

    for (size_t sample : sampleNumbers) {
        ALSVINN_LOG(INFO, "Running sample: " << sample << std::endl);
        alsfvm::init::Parameters parameters;

        for (auto parameterName : parameterNames) {
            auto samples = sampleGenerator->generate(parameterName, sample);
            parameters.addParameter(parameterName,
                samples);

        }

        auto simulator = simulatorCreator->createSimulator(parameters, sample);

        for ( auto& statisticWriter : statistics) {
            simulator->addWriter(std::dynamic_pointer_cast<alsfvm::io::Writer>
                (statisticWriter));

            auto timestepAdjuster =
                alsfvm::dynamic_pointer_cast<alsfvm::integrator::TimestepAdjuster>
                (statisticWriter);

            if (timestepAdjuster) {
                simulator->addTimestepAdjuster(timestepAdjuster);
            }
        }

        simulator->callWriters();

        while (!simulator->atEnd()) {
            simulator->performStep();
            timestepsPerformedTotal++;
        }

        simulator->finalize();
        grid = simulator->getGrid();

    }

    for (auto& statisticsWriter : statistics) {
        statisticsWriter->combineStatistics();

        if (mpiConfig->getRank() == 0) {
            statisticsWriter->finalizeStatistics();
            statisticsWriter->writeStatistics(*grid);
        }
    }


}


void Runner::setStatistics(const
    std::vector<std::shared_ptr<stats::Statistics> >& statistics) {
    this->statistics = statistics;
}

std::string Runner::getName() const {
    return name;
}

size_t Runner::getTimestepsPerformedTotal() const {
    return timestepsPerformedTotal;
}

}
}
