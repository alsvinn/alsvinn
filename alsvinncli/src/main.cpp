#include <alsfvm/config/SimulatorSetup.hpp>
#include <cmath>
int main(int argc, char** argv) {

    if (argc != 2) {
        std::cout << "Usage:\n\t" << argv[0] << " <inputfile.xml>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string inputfile = argv[1];

    alsfvm::config::SimulatorSetup setup;

    auto simulator = setup.readSetupFromFile(inputfile);

    std::cout << "Running simulator... " << std::endl;
    std::cout << std::endl << std::endl;
    std::cout << std::numeric_limits<long double>::digits10 + 1;

    simulator->callWriters();
    while(!simulator->atEnd()) {

        simulator->performStep();
        std::cout << "\rPercent done: " << std::round(100.0 * simulator->getCurrentTime() / simulator->getEndTime())  << std::flush;

    }


    std::cout << std::endl << std::endl;
    std::cout << "Simulation finished!" << std::endl;
    return EXIT_SUCCESS;
}
