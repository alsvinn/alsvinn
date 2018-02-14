#include "alsfvm/numflux/euler/HLL.hpp"
namespace alsfvm {
namespace numflux {
namespace euler {

template<>
const std::string HLL<3>::name = "hll";

template<>
const std::string HLL<2>::name = "hll";

template<>
const std::string HLL<1>::name = "hll";


} // namespace alsfvm
} // namespace numflux
} // namespace euler

