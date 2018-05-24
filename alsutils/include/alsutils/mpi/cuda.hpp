/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


//! Various utilities for mpi and cuda.


namespace alsutils {
namespace mpi {


//! Checks wether GPU direct is enabled or not
//!
//! We check according to this documentation
//!
//!    https://www.open-mpi.org/faq/?category=runcuda
//!
//! under
//!
//!   4. Can I tell at compile time or runtime whether I have CUDA-aware support?
bool hasGPUDirectSupport() {
#ifdef ALSVINN_HAS_GPU_DIRECT
    return true;
#else
    return false;
#endif
}
}
}
