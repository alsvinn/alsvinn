#include "alsuq/generator/STLMersenne.hpp"

namespace alsuq { namespace generator {

std::shared_ptr<Generator> STLMersenne::getInstance()
{
    static std::shared_ptr<Generator> instance;

    if (!instance) {
        instance.reset(new STLMersenne());
    }
    return instance;
}

real STLMersenne::generate(size_t component)
{
    return distribution(generator);
}

}
                }
