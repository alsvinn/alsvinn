#include "alsuq/generator/Well512A.hpp"
#include <array>
/* ***************************************************************************** */
/* Copyright:      Francois Panneton and Pierre L'Ecuyer, University of Montreal */
/*                 Makoto Matsumoto, Hiroshima University                        */
/* Notice:         This code can be used freely for personal, academic,          */
/*                 or non-commercial purposes. For commercial purposes,          */
/*                 please contact P. L'Ecuyer at: lecuyer@iro.UMontreal.ca       */
/* ***************************************************************************** */

#define W 32
#define R 16
#define P 0
#define M1 13
#define M2 9
#define M3 5

#define MAT0POS(t,v) (v^(v>>t))
#define MAT0NEG(t,v) (v^(v<<(-(t))))
#define MAT3NEG(t,v) (v<<(-(t)))
#define MAT4NEG(t,b,v) (v ^ ((v<<(-(t))) & b))

#define V0            STATE[state_i                   ]
#define VM1           STATE[(state_i+M1) & 0x0000000fU]
#define VM2           STATE[(state_i+M2) & 0x0000000fU]
#define VM3           STATE[(state_i+M3) & 0x0000000fU]
#define VRm1          STATE[(state_i+15) & 0x0000000fU]
#define VRm2          STATE[(state_i+14) & 0x0000000fU]
#define newV0         STATE[(state_i+15) & 0x0000000fU]
#define newV1         STATE[state_i                 ]
#define newVRm1       STATE[(state_i+14) & 0x0000000fU]

#define FACT 2.32830643653869628906e-10
namespace alsuq { namespace generator {

Well512A::Well512A()
{
    int seed = 0;

    std::array<unsigned int, R> buffer;

     // init buffer using Linear Congruential Generator

     const long long a = 1103515245;
     const long long c = 12345;

     unsigned int x = seed;

     for (int i=0; i<R; i++) {
       x = a*x + c;
       buffer[i] = x;
     }
     int j;
     state_i = 0;
     for (j = 0; j < R; j++) {
            STATE[j] = buffer[j];
     }
}

std::shared_ptr<Generator> Well512A::getInstance()
{
    static std::shared_ptr<Generator> instance;

    if (!instance) {
        instance.reset(new Well512A());
    }
    return instance;
}

real Well512A::generate(size_t component)
{
     z0    = VRm1;
     z1    = MAT0NEG (-16,V0)    ^ MAT0NEG (-15, VM1);
     z2    = MAT0POS (11, VM2)  ;
     newV1 = z1                  ^ z2;
     newV0 = MAT0NEG (-2,z0)     ^ MAT0NEG(-18,z1)    ^ MAT3NEG(-28,z2) ^ MAT4NEG(-5,0xda442d24U,newV1) ;
     state_i = (state_i + 15) & 0x0000000fU;
     return ((double) STATE[state_i]) * FACT;
}

}
}
