#pragma once
#include "alsuq/generator/Generator.hpp"
namespace alsuq {
namespace generator {

//! \note This class is a singleton
class Well512A : public Generator {
    public:
        //! Gets the one instance of the Well512 generator
        static std::shared_ptr<Generator> getInstance();

        //! Generates the next random number
        real generate(size_t component);
    private:
        // Singleton
        Well512A();

        static constexpr size_t R = 16;
        unsigned int state_i = 0;
        unsigned int STATE[R];
        unsigned int z0, z1, z2;
};
} // namespace generator
} // namespace alsuq
