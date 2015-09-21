#pragma once
#include "alsfvm/types.hpp"
namespace alsfvm {
	namespace equation {
		namespace euler {

			///
			/// Holds all the relevant views for the equation (extra variables)
			/// \note We template on VolumeType and ViewType to allow for const and non-const in one.
			/// \note We could potentially only template on one of these and use decltype, but there is a 
			/// bug in MS VC 2013 (http://stackoverflow.com/questions/21609700/error-type-name-is-not-allowed-message-in-editor-but-not-during-compile)
			///
			template<class VolumeType, class ViewType>
			class ViewsExtra {
			public:

				ViewsExtra(VolumeType& volume)
					: p(volume.getScalarMemoryArea("p")->getView()),
					ux(volume.getScalarMemoryArea("ux")->getView()),
					uy(volume.getScalarMemoryArea("uy")->getView()),
					uz(volume.getScalarMemoryArea("uz")->getView())
				{
					// Empty
				}

				template<size_t variableIndex>
				__device__ __host__ ViewType& get() {
					static_assert(variableIndex < 5, "We only have 5 conserved variables for Euler!");
					switch (variableIndex) {
					case 0:
						return p;
						break;
					case 1:
						return ux;
						break;
					case 2:
						return uy;
						break;
					case 3:
						return uz;
						break;
					}
					// If we reach this far, something has gone wrong
					assert(false);
				}

				__device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
					return p.index(x, y, z);
				}


				ViewType p;
				ViewType ux;
				ViewType uy;
				ViewType uz;
			};


		} // namespace alsfvm
	} // namespace equation
} // namespace euler
