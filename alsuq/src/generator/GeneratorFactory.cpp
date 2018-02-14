#include "alsuq/generator/GeneratorFactory.hpp"
#include "alsuq/generator/STLMersenne.hpp"
#include "alsuq/generator/Well512A.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq {
namespace generator {

std::shared_ptr<Generator> GeneratorFactory::makeGenerator(
    const std::string& name,
    const size_t dimensions,
    const size_t numberVariables
) {

    if (name == "stlmersenne") {
        return STLMersenne::getInstance();
    } else if (name == "well512a") {
        return Well512A::getInstance();
    } else {
        THROW("Unknown generator " << name);
    }


}

}
}
