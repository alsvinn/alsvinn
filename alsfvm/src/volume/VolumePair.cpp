#include "alsfvm/volume/VolumePair.hpp"

namespace alsfvm { namespace volume {

VolumePair::VolumePair(std::shared_ptr<Volume> conservedVolume,
                       std::shared_ptr<Volume> extraVolume)
    : volumes{conservedVolume, extraVolume}
{

}

std::shared_ptr<Volume> VolumePair::getConservedVolume()
{
    return volumes[0];
}

std::shared_ptr<Volume> VolumePair::getExtraVolume()
{
    return volumes[1];
}

VolumePair::IteratorType VolumePair::begin()
{
    return volumes.begin();
}

VolumePair::IteratorType VolumePair::end()
{
    return volumes.end();
}

VolumePair::ConstIteratorType VolumePair::begin() const
{
    return volumes.begin();
}

VolumePair::ConstIteratorType VolumePair::end() const
{
    return volumes.end();
}

}
}
