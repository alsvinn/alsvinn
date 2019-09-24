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

#include "alsfvm/memory/Memory.hpp"
namespace alsfvm {
namespace cuda {

template<class T>
class CudaMemory : public memory::Memory<T> {
public:
    ///
    /// \param nx the number of cells in x direction
    /// \param ny the number of cells in y direction
    /// \param nz the number of cells in z direction
    ///
    CudaMemory(size_t nx, size_t ny = 1, size_t nz = 1);


    //! Clones the memory area, but *does not copy the content*
    virtual std::shared_ptr<memory::Memory<T> > makeInstance() const override;

    // Note: Virtual distructor since we will inherit
    // from this.
    virtual ~CudaMemory();

    //! Copies the contents of the other memory area into this one
    virtual void copyFrom(const memory::Memory<T>& other) override;

    ///
    /// Checks if the memory area is on the host (CPU) or
    /// on some device, if the latter, one needs to copy to host
    /// before reading it.
    /// @returns false
    ///
    virtual bool isOnHost() const;

    ///
    /// Gets the pointer to the data (need not be on the host!)
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
    ///
    virtual void copyToHost(T* bufferPointer, size_t bufferLength) const override;


    ///
    /// Copies the memory from the buffer (assumed to be on Host/CPU)
    ///
    virtual void copyFromHost(const T* bufferPointer, size_t bufferLength) override;




    ///
    /// Adds the other memory area to this one
    /// \param other the memory area to add from
    ///
    virtual void operator+=(const memory::Memory<T>& other) override;

    ///
    /// Mutliplies the other memory area to this one
    /// \param other the memory area to multiply from
    ///
    virtual void operator*=(const memory::Memory<T>& other) override;

    ///
    /// Subtracts the other memory area to this one
    /// \param other the memory area to subtract from
    ///
    virtual void operator-=(const memory::Memory<T>& other) override;

    ///
    /// Divides the other memory area to this one
    /// \param other the memory area to divide from
    ///
    virtual void operator/=(const memory::Memory<T>& other) override;


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
    /// Sets every component to zero
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
    /// This calls cudaMemcpy3d behind the scenes.
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
        T a2, const memory::Memory<T>& v2,
        T a3, const memory::Memory<T>& v3,
        T a4, const memory::Memory<T>& v4,
        T a5, const memory::Memory<T>& v5) override;



    //! Adds a power of the other memory area to this memory area, ie
    //!
    //! \f[this += pow(other, power)\f]
    //!
    //! @param other the other memory area to the the power of
    //! @param power the power to use
    virtual void addPower(const memory::Memory<T>& other, double power) override;

    //! Adds a power of the other memory area to this memory area, ie
    //!
    //! \f[this += factor*pow(other, power)\f]
    //!
    //! @param other the other memory area to the the power of
    //! @param power the power to use
    virtual void addPower(const memory::Memory<T>& other, double power,
        double factor) override;


    //! Subtract a power of the other memory area to this memory area, ie
    //!
    //! \f[this -= pow(other, power)\f]
    //!
    //! @param other the other memory area to the the power of
    //! @param power the power to use
    virtual void subtractPower(const memory::Memory<T>& other,
        double power) override;


    std::shared_ptr<memory::Memory<T> > getHostMemory() override;

    const std::shared_ptr<const memory::Memory<T> > getHostMemory() const override;

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
    T* memoryPointer;
};

}
}
