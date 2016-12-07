#include "alsuq/generator/GeneratorFactory.hpp"
#include "alsuq/generator/Normal.hpp"
#include "alsuq/generator/Uniform.hpp"

namespace alsuq { namespace generator {

std::shared_ptr<Generator> makeGenerator(const std::string& name,
                                         const size_t dimensions,
                                         const size_t numberVariables,
                                         const Parameters& parameters)
{
    std::shared_ptr<Generator> generator;
    if (name == "uniform") {
        generator.reset(new Uniform(parameters));
    } else if (name == "normal") {
        generator.reset(new Uniform(parameters));
    } else {
        THROW("Unknown generator " << name);
    }
}

}
}
