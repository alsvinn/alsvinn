#include <alsuq/config/Setup.hpp>
#include <iostream>
#include <fstream>
#include <limits>
#include <mpi.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage:\n\t"<<argv[0] << " path/to/xml_file.xml output_basename" << std::endl;
    return 1;
  }
  MPI_Init(&argc, &argv);
  std::string inputFilename = argv[1];
  std::string outputFilename = argv[2];
  alsuq::config::Setup setup;

  auto sampleGenerator = setup.makeSampleGenerator(inputFilename);

  auto numberOfSamples = setup.readNumberOfSamples(inputFilename);

  auto sampleStart = setup.readSampleStart(inputFilename);

  for (const auto& parameterName : sampleGenerator->getParameterList()) {
    std::ofstream output(outputFilename + "_" + parameterName + ".txt");
    output.precision(std::numeric_limits<double>::max_digits10+1);
    for (size_t sample = 0; sample < numberOfSamples; ++sample) {
      auto parameter = sampleGenerator->generate(parameterName, sample + sampleStart);

      for (auto component : parameter) {
	output << component << " ";
      }
      output << "\n";
    }
  }

      


      

      
  return 0;
}
