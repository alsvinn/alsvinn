#include "alsfvm/reconstruction/ENOCoefficients.hpp"

namespace alsfvm { namespace reconstruction {

    template<>
    real ENOCoeffiecients<1>::coefficients[][1] =
    {
        {1},
        {1}
    };

    template<>
    real ENOCoeffiecients<2>::coefficients[][2] =
    {
        {3.0/2.0, -1.0/2.0},
        {1.0/2.0, 1.0/2.0},
        {-1.0/2.0, 3.0/2.0}
    };

    template<>
    real ENOCoeffiecients<3>::coefficients[][3] =
    {
        {11.0/6.0, -7.0/6.0, 1.0/3.0},
        {1.0/3.0, 5.0/6.0, -1.0/6.0},
        {-1.0/6.0, 5.0/6.0, 1.0/3.0},
        {1.0/3.0, -7.0/6.0, 11.0/6.0}
    };

    template<>
    real ENOCoeffiecients<4>::coefficients[][4] =
    {
        {25.0/12.0,  -23.0/12.0,  13.0/12.0, -1.0/4.0},
        { 1.0/4.0,    13.0/12.0,  -5.0/12.0,  1.0/12.0},
        {-1.0/12.0,    7.0/12.0,   7.0/12.0, -1.0/12.0},
        { 1.0/12.0,   -5.0/12.0,  13.0/12.0,  1.0/4.0},
        {-1.0/4.0,    13.0/12.0, -23.0/12.0, 25.0/12.0}
    };

}}

