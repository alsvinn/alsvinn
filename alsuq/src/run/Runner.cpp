#include "alsuq/run/Runner.hpp"

#include "alsutils/log.hpp"

namespace alsuq { namespace run {

Runner::Runner(std::shared_ptr<SimulatorCreator> simulatorCreator,
               std::shared_ptr<samples::SampleGenerator> sampleGenerator,
               std::vector<size_t> sampleNumbers)
    : simulatorCreator(simulatorCreator),
      sampleGenerator(sampleGenerator),
      parameterNames(sampleGenerator->getParameterList()),
      sampleNumbers(sampleNumbers)

{

}

void Runner::run()
{
    std::shared_ptr<alsfvm::grid::Grid> grid;
    for(size_t sample : sampleNumbers) {
        ALSVINN_LOG(INFO, "Running sample: " << sample << std::endl);
        alsfvm::init::Parameters parameters;

        for (auto parameterName : parameterNames) {
            auto samples = sampleGenerator->generate(parameterName, sample);
            parameters.addParameter(parameterName,
                                    samples);

        }

        auto simulator = simulatorCreator->createSimulator(parameters, sample);
        for( auto & statisticWriter : statistics) {
            simulator->addWriter(std::dynamic_pointer_cast<alsfvm::io::Writer>(statisticWriter));

            auto timestepAdjuster = alsfvm::dynamic_pointer_cast<alsfvm::integrator::TimestepAdjuster>(statisticWriter);
            if (timestepAdjuster) {
                simulator->addTimestepAdjuster(timestepAdjuster);
            }
        }
        simulator->callWriters();
        while (!simulator->atEnd()) {
            simulator->performStep();
        }
        grid = simulator->getGrid();

    }

    for(auto& statisticsWriter : statistics) {
        statisticsWriter->combineStatistics();

        if (mpiConfig.getRank() == 0) {
            statisticsWriter->finalize();
            statisticsWriter->writeStatistics(*grid);
        }
    }


}


void Runner::setStatistics(const std::vector<std::shared_ptr<stats::Statistics> >& statistics)
{
    this->statistics = statistics;
}

}
                }
