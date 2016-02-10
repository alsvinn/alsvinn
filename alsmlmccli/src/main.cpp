#include <alsfvm/config/SimulatorSetup.hpp>
#include <cmath>
#include <boost/chrono.hpp>
#include <boost/program_options.hpp>
#include <thread>
#include "alsfvm/io/QueueWriter.hpp"
#include "alsfvm/io/FixedIntervalWriter.hpp"


using namespace alsfvm::io;
using namespace alsfvm;
using namespace alsfvm::volume;
using namespace alsfvm::memory;

int main(int argc, char** argv) {
	try {
        using namespace boost::program_options;
        options_description description;

        description.add_options()
                ("help", "Produces this help message")
                ("coarse-file,c", value<std::string>(), "the coarse file to use")
                ("fine-file,f", value<std::string>(), "the fine file to use")
                ("stabilization-time,s", value<double>(), "the stabilization time to use");



        variables_map vm;
        store(parse_command_line(argc, argv, description), vm);
        notify(vm);

        if (vm.count("help")) {
            std::cout << description << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        if (vm.count("coarse-file") == 0 || vm.count("fine-file") == 0
                || vm.count("stabilization-time") == 0 ) {
            std::cout << "Wrong arguments supplied." << std::endl;
            std::cout << description << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::string inputCoarse = vm["coarse-file"].as<std::string>();
        std::string inputFine = vm["fine-file"].as<std::string>();
        double stabilizationTime = vm["stabilization-time"].as<double>();



		alsfvm::config::SimulatorSetup setup;

        auto simulatorCoarse = setup.readSetupFromFile(inputCoarse);
        auto simulatorFine = setup.readSetupFromFile(inputFine);


        auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>(simulatorCoarse->getPlatformName());
        auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
        auto volumeFactory = alsfvm::make_shared<VolumeFactory>(simulatorCoarse->getEquationName(), memoryFactory);
        auto queueWriter = alsfvm::make_shared<QueueWriter>(1, volumeFactory);

        auto queueWriterAsWriter = alsfvm::dynamic_pointer_cast<Writer>(queueWriter);
        alsfvm::shared_ptr<io::Writer> intervalWriter(
                    new FixedIntervalWriter(queueWriterAsWriter, stabilizationTime, 0));

        simulatorFine->addWriter(intervalWriter);
        auto timestepAdjuster = alsfvm::dynamic_pointer_cast<integrator::TimestepAdjuster>(intervalWriter);

        simulatorFine->addTimestepAdjuster(timestepAdjuster);
        simulatorCoarse->addTimestepAdjuster(timestepAdjuster);

        std::thread fineThread([&] () {
            try {
                simulatorFine->callWriters();
                while (!simulatorFine->atEnd()) {
                    simulatorFine->performStep();
                }
            } catch(std::runtime_error& e) {
                std::cerr << "Error!" << std::endl;
                std::cerr << e.what() << std::endl;
            }
        });


        std::thread coarseThread([&] () {
            try {
                // note: We count the timestep 0 as the first stabilization
                size_t stabilizationsDone = 1;
                while (!simulatorCoarse->atEnd()) {
                    simulatorCoarse->performStep();

                    if (simulatorCoarse->getCurrentTime() >= stabilizationsDone * stabilizationTime) {
                        queueWriter->pop([&](const volume::Volume& volume) {
                            std::cout <<"Updating coarse simulation state at time " << simulatorCoarse->getCurrentTime() << std::endl;
                            simulatorCoarse->setSimulationState(volume);
                        });

                        stabilizationsDone++;
                    }
                }
            } catch(std::runtime_error& e) {
                std::cerr << "Error!" << std::endl;
                std::cerr << e.what() << std::endl;
            }

        });

        coarseThread.join();
        fineThread.join();

		std::cout << "Simulation finished!" << std::endl;
	}
	catch (std::runtime_error& e) {
		std::cerr << "Error!" << std::endl;
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
    return EXIT_SUCCESS;
}
