#include "alsfvm/numflux/euler/HLL3.hpp"
namespace alsfvm {
namespace numflux {
namespace euler {


template<>
const std::string HLL3<3>::name = "hll3";

template<>
const std::string HLL3<2>::name = "hll3";

template<>
const std::string HLL3<1>::name = "hll3";
} // namespace alsfvm
} // namespace numflux
} // namespace euler

