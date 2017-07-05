#pragma once
#include "alsfvm/types.hpp"
#include <cassert>

namespace alsfvm {
	namespace equation {
		namespace euler {

			///
			/// Holds all the relevant views for the equation (extra variables)
			/// \note We template on VolumeType and ViewType to allow for const and non-const in one.
			/// \note We could potentially only template on one of these and use decltype, but there is a 
			/// bug in MS VC 2013 (http://stackoverflow.com/questions/21609700/error-type-name-is-not-allowed-message-in-editor-but-not-during-compile)
			///
			template<class VolumeType, class ViewType, int nsd>
            class ViewsExtra {
            public:

            };

            template<class VolumeType, class ViewType>
            class ViewsExtra<VolumeType, ViewType, 3> {
			public:
                typedef typename Types<3>::rvec rvec;
                typedef typename std::conditional<std::is_const<VolumeType>::value,
                const real&,
                real&>::type reference_type;

                typedef typename Types<3>::template vec<reference_type> reference_vec;
                
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
					case 1:
						return ux;
					case 2:
						return uy;
					case 3:
						return uz;
					}
					// If we reach this far, something has gone wrong
					assert(false);
                    return p;
				}

				__device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
					return p.index(x, y, z);
				}



                __device__ __host__ reference_vec u(size_t index)  {
                    return reference_vec(ux.at(index), uy.at(index), uz.at(index));
                }

                __device__ __host__ rvec u(size_t index) const {
                    return rvec(ux.at(index), uy.at(index), uz.at(index));
                }


             
				ViewType p;
				ViewType ux;
				ViewType uy;
				ViewType uz;
			};

            template<class VolumeType, class ViewType>
            class ViewsExtra<VolumeType, ViewType, 2> {
            public:
                typedef typename Types<2>::rvec rvec;
                typedef typename std::conditional<std::is_const<VolumeType>::value,
                const real&,
                real&>::type reference_type;

                typedef typename Types<2>::template vec<reference_type> reference_vec;


                ViewsExtra(VolumeType& volume)
                    : p(volume.getScalarMemoryArea("p")->getView()),
                    ux(volume.getScalarMemoryArea("ux")->getView()),
                    uy(volume.getScalarMemoryArea("uy")->getView())
                {
                    // Empty
                }

                template<size_t variableIndex>
                __device__ __host__ ViewType& get() {
                    static_assert(variableIndex < 4, "We only have 5 conserved variables for Euler!");
                    switch (variableIndex) {
                    case 0:
                        return p;
                    case 1:
                        return ux;
                    case 2:
                        return uy;
                    }
                    // If we reach this far, something has gone wrong
                    assert(false);
                    return p;
                }

                __device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
                    return p.index(x, y, z);
                }

                __device__ __host__ reference_vec u(size_t index) {
                    return reference_vec(ux.at(index), uy.at(index));
                }

                __device__ __host__ rvec u(size_t index) const {
                    return rvec(ux.at(index), uy.at(index));
                }


                ViewType p;
                ViewType ux;
                ViewType uy;
            };


            template<class VolumeType, class ViewType>
            class ViewsExtra<VolumeType, ViewType, 1> {
            public:
                typedef typename Types<1>::rvec rvec;
                typedef typename std::conditional<std::is_const<VolumeType>::value,
                const real&,
                real&>::type reference_type;

                typedef typename Types<1>::template vec<reference_type> reference_vec;


                ViewsExtra(VolumeType& volume)
                    : p(volume.getScalarMemoryArea("p")->getView()),
                    ux(volume.getScalarMemoryArea("ux")->getView())
                {
                    // Empty
                }

                template<size_t variableIndex>
                __device__ __host__ ViewType& get() {
                    static_assert(variableIndex < 2, "We only have 5 conserved variables for Euler!");
                    switch (variableIndex) {
                    case 0:
                        return p;
                    case 1:
                        return ux;
                    }
                    // If we reach this far, something has gone wrong
                    assert(false);
                    return p;
                }

                __device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
                    return p.index(x, y, z);
                }

                __device__ __host__ reference_vec u(size_t index) {
                    return reference_vec(ux.at(index));
                }

                __device__ __host__ rvec u(size_t index) const {
                    return rvec(ux.at(index));
                }

                ViewType p;
                ViewType ux;
            };


		} // namespace alsfvm
	} // namespace equation
} // namespace euler
