#pragma once
#include <cassert>
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
                        product(row, column) += (*this)(row, i)*matrix(i, column);
                    }
                }
            }

            return product;
        }

        __device__ __host__ matrix<T, NumberOfColumns, NumberOfRows> transposed() const {
            matrix<T, NumberOfColumns, NumberOfRows> transposedMatrix;
            for (size_t column = 0; column < NumberOfColumns; ++column) {
                for (size_t row = 0; row < NumberOfRows; ++row) {
                    transposedMatrix(row, column) = (*this)(column, row);
                }
            }

            return transposedMatrix;
        }

        __device__ __host__ matrix<T, NumberOfColumns, NumberOfRows> normalized() const {
            matrix<T, NumberOfColumns, NumberOfRows> newMatrix;
            for (size_t column = 0; column < NumberOfColumns; ++column) {
                T norm = 0;
                for (size_t row = 0; row < NumberOfRows; ++row) {
                    norm += (*this)(row, column)*(*this)(row, column);
                }
                norm = sqrt(norm);
                for (size_t row = 0; row < NumberOfRows; ++row) {
                    newMatrix(row, column) = (*this)(row, column) /norm;
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

       
    private:
        T data[NumberOfColumns][NumberOfRows];
    };


}

template<class T, size_t NumberOfRows, size_t NumberOfColumns>
inline std::ostream& operator<<(std::ostream& os, const alsutils::matrix<T, NumberOfRows, NumberOfColumns>& mat) {
    os << "[" << std::endl;
    for (int i = 0; i < NumberOfRows; ++i) {
        for (int j = 0; j < NumberOfColumns; ++j) {
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


