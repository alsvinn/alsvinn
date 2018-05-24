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
#include <cassert>
#include <iostream>
#include <string>
namespace alsutils {

template<class T, size_t NumberOfRows, size_t NumberOfColumns>
class matrix {
public:

    typedef matrix<T, NumberOfRows, NumberOfColumns> self_type;
    //! Creates a matrix initialized to 0.
    __device__ __host__  matrix() {

        for (size_t column = 0; column < NumberOfColumns; ++column) {
            for (size_t row = 0; row < NumberOfRows; ++row) {
                data[column][row] = T(0);
            }
        }
    }

    //! Get the matrix element at the given row and column.
    __device__ __host__ const T& operator()(size_t row, size_t column) const {
        assert(row < NumberOfRows);
        assert(column < NumberOfColumns);
        return data[column][row];
    }

    //! Get the matrix element at the given row and column.
    __device__ __host__ T& operator()(size_t row, size_t column) {
        assert(row < NumberOfRows);
        assert(column < NumberOfColumns);
        return data[column][row];
    }


    //! Matrix-vector multiplication. We only support this for
    //! quadratic matrices as of now.
    template<class VectorType>
    __device__ __host__ VectorType operator*(const VectorType& vector) const {
        static_assert(NumberOfColumns == NumberOfRows,
            "Matrix-Vector multiplication only supported for quadratic matrices.");
        static_assert(VectorType::size() == NumberOfColumns,
            "Matrix vector multiplication given wrong dimensions");

        VectorType product;

        for (size_t column = 0; column < NumberOfColumns; ++column) {
            for (size_t row = 0; row < NumberOfRows; ++row) {
                product[row] += (*this)(row, column) *  vector[column];
            }
        }

        return product;
    }


    __device__ __host__ self_type operator*(const self_type& matrix) const {
        static_assert(NumberOfColumns == NumberOfRows,
            "Matrix-Matrix multiplication only supported for quadratic matrices.");

        self_type product;

        for (size_t row = 0; row < NumberOfRows; ++row) {
            for (size_t column = 0; column < NumberOfColumns; ++column) {
                for (size_t i = 0; i < NumberOfRows; ++i) {
                    product(row, column) += (*this)(row, i) * matrix(i, column);
                }
            }
        }

        return product;
    }

    __device__ __host__ matrix<T, NumberOfColumns, NumberOfRows> transposed()
    const {
        matrix<T, NumberOfColumns, NumberOfRows> transposedMatrix;

        for (size_t column = 0; column < NumberOfColumns; ++column) {
            for (size_t row = 0; row < NumberOfRows; ++row) {
                transposedMatrix(column, row) = (*this)(row, column);
            }
        }

        return transposedMatrix;
    }

    __device__ __host__ matrix<T, NumberOfColumns, NumberOfRows> normalized()
    const {
        matrix<T, NumberOfColumns, NumberOfRows> newMatrix;

        for (size_t column = 0; column < NumberOfColumns; ++column) {
            T norm = 0;

            for (size_t row = 0; row < NumberOfRows; ++row) {
                norm += (*this)(row, column) * (*this)(row, column);
            }

            norm = sqrtf(norm);

            for (size_t row = 0; row < NumberOfRows; ++row) {
                newMatrix(row, column) = (*this)(row, column) / norm;
            }
        }

        return newMatrix;
    }

    static __device__ __host__ matrix<T, NumberOfColumns, NumberOfRows> identity() {
        static_assert(NumberOfColumns == NumberOfRows,
            "Matrix-Vector multiplication only supported for quadratic matrices.");
        matrix<T, NumberOfColumns, NumberOfRows> identityMatrix;

        for (size_t i = 0; i < NumberOfColumns; ++i) {
            identityMatrix(i, i) = 1;
        }

        return identityMatrix;
    }

    __host__ std::string str() const {
        std::stringstream ss;

        for (size_t i = 0; i < NumberOfRows; ++i) {
            for (size_t j = 0; j < NumberOfColumns; ++j) {
                ss << (*this)(i, j);

                if (j < NumberOfColumns - 1) {
                    ss << ", ";
                }
            }
        }

        return ss.str();
    }


private:
    T data[NumberOfColumns][NumberOfRows];
};


}

template<class T, size_t NumberOfRows, size_t NumberOfColumns>
std::ostream& operator<<(
    std::ostream& os,
    const alsutils::matrix<T, NumberOfRows, NumberOfColumns>& mat) {
    os << "[" << std::endl;

    for (size_t i = 0; i < NumberOfRows; ++i) {
        for (size_t j = 0; j < NumberOfColumns; ++j) {
            os << mat(i, j);

            if (j < NumberOfColumns - 1) {
                os << ", ";
            }
        }

        os << std::endl;
    }

    os << "]";
    return os;
}





