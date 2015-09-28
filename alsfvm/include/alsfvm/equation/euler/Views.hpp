#pragma once
#include "alsfvm/types.hpp"
#include <cassert>
namespace alsfvm { namespace equation { namespace euler { 

	///
	/// Holds all the relevant views for the equation.
	/// \note We template on VolumeType and ViewType to allow for const and non-const in one.
	/// \note We could potentially only template on one of these and use decltype, but there is a 
	/// bug in MS VC 2013 (http://stackoverflow.com/questions/21609700/error-type-name-is-not-allowed-message-in-editor-but-not-during-compile)
	///
	template<class VolumeType, class ViewType>
    class Views {
    public:
		
		Views(VolumeType& volume) 
			: rho(volume.getScalarMemoryArea("rho")->getView()), 
			mx(volume.getScalarMemoryArea("mx")->getView()),
			my(volume.getScalarMemoryArea("my")->getView()),
			mz(volume.getScalarMemoryArea("mz")->getView()),
			E(volume.getScalarMemoryArea("E")->getView()) {
			// Empty
		}

		template<size_t variableIndex>
		__device__ __host__ ViewType& get() {
			static_assert(variableIndex < 5, "We only have 5 conserved variables for Euler!");
			switch (variableIndex) {
			case 0: 
				return rho;
				break;
			case 1:
				return mx;
				break;
			case 2:
				return my;
				break;
			case 3:
				return mz;
				break;
			case 4:
				return E;
				break;
			}
			// If we reach this far, something has gone wrong
			assert(false);
		}

		__device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
			return rho.index(x, y, z);
		}

	
		ViewType rho;
		ViewType mx;
		ViewType my;
		ViewType mz;
		ViewType E;
    };

	
} // namespace alsfvm
} // namespace equation
} // namespace euler
