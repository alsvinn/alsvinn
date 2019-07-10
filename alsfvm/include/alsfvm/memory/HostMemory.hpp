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
#include <vector>
#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/memory/index.hpp"

namespace alsfvm {
namespace memory {

template<class T>
class HostMemory : public alsfvm::memory::Memory<T> {
public:

    ///
    /// @param nx the size of the memory area in X (number of T)
    /// @param ny the size of the memory area in Y (number of T)
    /// @param nz the size of the memory area in Z (number of T)
    ///
    HostMemory(size_t nx, size_t ny = 1, size_t nz = 1);


    //! Clones the memory area, but *does not copy the content*
    virtual std::shared_ptr<Memory<T> > makeInstance() const override;

    ///
    /// Checks if the memory area is on the host (CPU) or
    /// on some device, if the latter, one needs to copy to host
    /// before reading it.
    /// @returns true if the memory is on host, false otherwise
    ///
    virtual bool isOnHost() const override;

    //! Copies the contents of the other memory area into this one
    virtual void copyFrom(const Memory<T>& other) override;

    ///
    /// Gets the pointer to the data (need not be on the host!)
    /// \note If this is an OpenCL implementation, the pointer will
    /// be useless! If you want to use the OpenCL memory, you should
    /// first cast to OpenCL memory, then get the OpenCL buffer pointer.
    ///
    virtual T* getPointer() override;

    ///
    /// Gets the pointer to the data (need not be on the host!)
    /// \note If this is an OpenCL implementation, the pointer will
    /// be useless! If you want to use the OpenCL memory, you should
    /// first cast to OpenCL memory, then get the OpenCL buffer pointer.
    ///
    virtual const T* getPointer() const override;

    ///
    /// Copies the memory to the given buffer
    /// \note bufferLength must be at least getSize()
    /// \param bufferPointer the buffer to write to
    /// \param bufferLength the size of the buffer (in number of T's)
    ///
    virtual void copyToHost(T* bufferPointer,
        size_t bufferLength) const override;

    ///
    /// Copies the memory from the buffer (assumed to be on Host/CPU)
    /// \note bufferLength must be at least getSize()
    /// \param bufferPointer the buffer to write to
    /// \param bufferLength the size of the buffer (in number of T's)
    ///
    virtual void copyFromHost(const T* bufferPointer,
        size_t bufferLength) override;


    ///
    /// Adds the other memory area to this one
    /// \param other the memory area to add from
    ///
    virtual void operator+=(const Memory<T>& other) override;

    ///
    /// Mutliplies the other memory area to this one
    /// \param other the memory area to multiply from
    ///
    virtual void operator*=(const Memory<T>& other) override;

    ///
    /// Subtracts the other memory area to this one
    /// \param other the memory area to subtract from
    ///
    virtual void operator-=(const Memory<T>& other) override;

    ///
    /// Divides the other memory area to this one
    /// \param other the memory area to divide from
    ///
    virtual void operator/=(const Memory<T>& other) override;

    ///
    /// Adds the scalar to each component
    /// \param scalar the scalar to add
    ///
    virtual void operator+=(real scalar) override;

    ///
    /// Multiplies the scalar to each component
    /// \param scalar the scalar to multiply
    ///
    virtual void operator*=(real scalar) override;

    ///
    /// Subtracts the scalar from each component
    /// \param scalar the scalar to subtract
    ///
    virtual void operator-=(real scalar) override;

    ///
    /// Divides the each component by the scalar
    /// \param scalar the scalar to divide
    ///
    virtual void operator/=(real scalar) override;

    ///
    /// \brief at returns the data at the given index
    /// \param x the x index
    /// \param y the y index
    /// \param z the z index
    /// \return the data at the given index
    ///
    T& at(size_t x, size_t y = 0, size_t z = 0);


    ///
    /// \brief at returns the data at the given index
    /// \param x the x index
    /// \param y the y index
    /// \param z the z index
    /// \return the data at the given index
    ///
    const T& at(size_t x, size_t y = 0, size_t z = 0) const;

    ///
    /// \brief makeZero sets every element to zero (0)
    ///
    virtual void makeZero() override;

    ///
    /// \brief copyInternalCells copies the internal cells into the memory area
    /// This is ideal for removing ghost cells before outputing the solution.
    /// \param startX start index (inclusive) for x direction
    /// \param endX end index (exclusive) for x direction
    /// \param startY start index (inclusive) for y direction
    /// \param endY end index (exclusive) for y direction
    /// \param startZ start index (inclusive) for z direction
    /// \param endZ end index (exclusive) for z direction
    /// \param output the output buffer
    /// \param outputSize must be at least the size of the written memory
    ///
    /// This is essentially equivalent to doing
    /// \code{.cpp}
    /// size_t numberOfZ = endZ-startZ;
    /// size_t numberOfY = endY-startY;
    /// size_t numberOfX = endX-startX;
    /// for(size_t z = startZ; z < endZ; z++) {
    ///     for(size_t y = startY; y < endY; y++) {
    ///         for(size_t x = startX; x < endX; x++) {
    ///             size_t indexIn = z * nx * ny + y * nx + x;
    ///             size_t indexOut = (z-startZ) * numberOfX * numberOfY
    ///                   + (y - startY) * numberOfY + (x - startX);
    ///             output[indexOut] = data[indexIn];
    ///          }
    ///     }
    /// }
    /// \endcode
    ///
    virtual void copyInternalCells(size_t startX, size_t endX,
        size_t startY, size_t endY,
        size_t startZ, size_t endZ,
        T* output, size_t outputSize) override;

    //! Adds the memory with coefficients to this memory area
    //! Here we compute the sum
    //! \f[ v_1^{\mathrm{new}}=a_1v_1+a_2v_2+a_3v_3+a_4v_4+a_5v_5+a_6v_6\f]
    //! where \f$v_1\f$ is the volume being operated on.
    virtual void addLinearCombination(T a1,
        T a2, const Memory<T>& v2,
        T a3, const Memory<T>& v3,
        T a4, const Memory<T>& v4,
        T a5, const Memory<T>& v5) override;


    //! Adds a power of the other memory area to this memory area, ie
    //!
    //! \f[this += pow(other, power)\f]
    //!
    //! @param other the other memory area to the the power of
    //! @param power the power to use
    virtual void addPower(const Memory<T>& other, double power) override;

    //! Subtracts a power of the other memory area to this memory area, ie
    //!
    //! \f[this -= pow(other, power)\f]
    //!
    //! @param other the other memory area to the the power of
    //! @param power the power to use
    virtual void subtractPower(const Memory<T>& other, double power) override;


    virtual std::shared_ptr<Memory<T> > getHostMemory() override;

    //! Copies the data to host if it is on GPU, otherwise makes a copy
    //! Const version
    virtual const std::shared_ptr<const Memory<T> > getHostMemory() const override;

    //! Computes the total variation, given here as
    //!
    //! \f[\sum_{i,j,k} \sqrt(\sum_{n=1}^d|u_{(i,j,k)}-u_{(i,j,k)-e_n}|^2)^p.\f]
    //!
    //! \note This function gives no performance guarantees
    //!
    //! @param p the exponent p
    //! @param start the index to start at (inclusive)
    //! @param end the maximum index (exclusive)
    virtual real getTotalVariation(int p, const ivec3& start,
        const ivec3& end) const override;

    //! Computes the total variation in a given direction \$d\in\{0,1,2\}\$
    //!
    //! \f[\sum_{i,j,k} |u_{(i,j,k)}-u_{(i,j,k)-e_n}|^p.\f]
    //!
    //! \note This function gives no performance guarantees
    //!
    //! @param p the exponent p
    //! @param direction the direction (between 0 and 2 inclusive)
    //! @param start the index to start at (inclusive)
    //! @param end the maximum index (exclusive)
    virtual real getTotalVariation(int direction, int p, const ivec3& start,
        const ivec3& end) const override;


private:
    std::vector<T> data;
};

template<class T>
T& HostMemory<T>::at(size_t x, size_t y, size_t z) {
    return data[calculateIndex(x, y, z, this->nx, this->ny)];
}

template<class T>
const T& HostMemory<T>::at(size_t x, size_t y, size_t z) const {
    return data[calculateIndex(x, y, z, this->nx, this->ny)];
}

} // namespace memory
} // namespace alsfvm


