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
#include <iostream>
#include "alsutils/error/Exception.hpp"
//! Executes the given code and checks for cuda error
#define CUDA_SAFE_CALL(x) { \
    cudaError_t error = x; \
    if (error != cudaSuccess) { \
        std::cerr << "Noticed CUDA error in " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "\tLine was:\"" << #x << "\"" << std::endl; \
        std::cerr << "\tError: " << cudaGetErrorString(error) << std::endl; \
        THROW("CUDA error" << std::endl << "Line was: " << std::endl <<"\t" << #x << "\nError code: " << error); \
    } \
}

//! Does the same as CUDA_SAFE_CALL, but doesn't print an error message
#define CUDA_SAFE_CALL_SILENT(x) { \
    cudaError_t error = x; \
    if (error != cudaSuccess) { \
        THROW("CUDA error" << std::endl << "Line was: " << std::endl <<"\t" << #x << "\nError code: " << error); \
    } \
}
