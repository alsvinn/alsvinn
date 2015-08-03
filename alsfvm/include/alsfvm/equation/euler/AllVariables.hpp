#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
#include <cassert>
#include <cmath>
namespace alsfvm { namespace equation { namespace euler { 
    class AllVariables : public ConservedVariables, public ExtraVariables {
    public:
		AllVariables(real rho, real mx, real my, real mz, real E, real p, real ux, real uy, real uz) :
			ConservedVariables(rho, mx, my, mz, E), ExtraVariables(p, ux, uy, uz)
		{
            assert(!std::isnan(rho));
            assert(!std::isnan(mx));
            assert(!std::isnan(my));
            assert(!std::isnan(mz));
            assert(!std::isnan(E));
            assert(!std::isnan(p));
            assert(!std::isnan(ux));
            assert(!std::isnan(uy));
            assert(!std::isnan(uz));
		}

		const ConservedVariables& conserved() const {
			return *this;
		}

        AllVariables(const ConservedVariables& conserved, const ExtraVariables& extra)
            : ConservedVariables(conserved), ExtraVariables(extra)
        {
            assert(!std::isnan(rho));
            assert(!std::isnan(m.x));
            assert(!std::isnan(m.y));
            assert(!std::isnan(m.z));
            assert(!std::isnan(E));
            assert(!std::isnan(p));
            assert(!std::isnan(u.x));
            assert(!std::isnan(u.y));
            assert(!std::isnan(u.z));
        }
		
    };


    ///
    /// Computes the component difference
    /// \note Makes a new instance
    ///
    inline AllVariables operator-(const AllVariables& a, const AllVariables& b) {
        assert(false);
    }

    ///
    /// Computes the component addition
    /// \note Makes a new instance
    ///
    inline AllVariables operator+(const AllVariables& a, const AllVariables& b) {
        assert(false);
    }

    ///
    /// Computes the product of a and b (scalar times vector)
    /// \note Makes a new instance
    ////
    inline AllVariables operator*(real a, const AllVariables& b) {
        assert(false);
    }

    ///
    /// Computes the division of a by b
    /// \note Makes a new instance
    ////
    inline AllVariables operator/(const AllVariables& a, real b) {
        assert(false);
    }

} // namespace alsfvm
} // namespace numflux
} // namespace euler

