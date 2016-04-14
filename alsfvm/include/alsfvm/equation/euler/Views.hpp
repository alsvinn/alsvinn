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

            if (variableIndex == 0) {
				return rho;
            }
            else if (variableIndex == 1) {
				return mx;
            }
            else if (variableIndex == 2) {
				return my;
            }
            else if (variableIndex == 3) {
				return mz;

            }
            else if (variableIndex == 4) {
				return E;
			}
			// If we reach this far, something has gone wrong
			assert(false);
            return rho;
		}


		__device__ __host__ ViewType& get(size_t variableIndex) {
			switch (variableIndex) {
			case 0:
				return rho;
			case 1:
				return mx;
			case 2:
				return my;
			case 3:
                return mz;
            case 4:
				return E;
			}
			// If we reach this far, something has gone wrong
			assert(false);
            return rho;
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
