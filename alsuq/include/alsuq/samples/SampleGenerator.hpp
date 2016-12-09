#pragma once
#include "alsuq/types.hpp"
#include "alsuq/generator/Generator.hpp"
#include "alsuq/distribution/Distribution.hpp"
#include <map>
#include <string>

namespace alsuq { namespace samples { 

    class SampleGenerator {
    public:
        typedef  std::map<std::string, std::pair<size_t,
        std::pair<
            std::shared_ptr<generator::Generator>,
            std::shared_ptr<distribution::Distribution> > > >
                GeneratorDistributionMap;


        SampleGenerator(const GeneratorDistributionMap& generators);


        std::vector<real> generate(const std::string& parameter, const size_t sampleIndex);

        std::vector<std::string> getParameterList() const;
    private:
        size_t currentSample{0};

        GeneratorDistributionMap generators;
    };
} // namespace samples
} // namespace alsuq
