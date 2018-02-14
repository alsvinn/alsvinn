#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
namespace volume {

//! Easy reference to the combination of conserved volume and extra volume
class VolumePair {
    public:
        VolumePair() {}
        typedef std::array<std::shared_ptr<volume::Volume>, 2>::iterator IteratorType;
        typedef std::array<std::shared_ptr<volume::Volume>, 2>::const_iterator
        ConstIteratorType;

        VolumePair(std::shared_ptr<volume::Volume> conservedVolume,
            std::shared_ptr<volume::Volume> extraVolume);

        std::shared_ptr<volume::Volume> getConservedVolume();
        std::shared_ptr<volume::Volume> getExtraVolume();

        IteratorType begin();
        IteratorType end();


        ConstIteratorType begin() const;
        ConstIteratorType end() const;
    private:
        std::array<std::shared_ptr<volume::Volume>, 2> volumes;

};
} // namespace volume
} // namespace alsfvm
