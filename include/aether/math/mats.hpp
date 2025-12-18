#pragma once
#include <cmath>
#include <array>
#include <aether/core/config.hpp>

namespace aether::math {

    // ======================================================
    // Vector temp
    // ======================================================
    template<int Size>
    struct Vec {
        std::array<double, Size> data{};
        static constexpr int size = Size;

        Vec() = default;

        // Construct from array
        explicit Vec(const double (&vals)[Size]) {
            #pragma omp simd
            for (int c = 0; c < Size; ++c) {
                data[c] = vals[c];
            }
        }

        // Element access
        AETHER_INLINE double& operator[](int i)             { return data[i]; }
        AETHER_INLINE const double& operator[](int i) const { return data[i]; }

        // Arithmetic Operators 
        AETHER_INLINE Vec operator+(const Vec& o) const {
            Vec result;
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                result.data[c] = data[c] + o.data[c];
            return result;
        }

        AETHER_INLINE Vec operator-(const Vec& o) const {
            Vec result;
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                result.data[c] = data[c] - o.data[c];
            return result;
        }

        AETHER_INLINE Vec operator*(const Vec& o) const {
            Vec result;
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                result.data[c] = data[c] * o.data[c];
            return result;
        }

        AETHER_INLINE Vec operator/(const Vec& o) const {
            Vec result;
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                result.data[c] = data[c] / o.data[c];
            return result;
        }

        // Scalar ops
        AETHER_INLINE Vec operator*(double s) const {
            Vec result;
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                result.data[c] = data[c] * s;
            return result;
        }

        AETHER_INLINE Vec operator/(double s) const {
            Vec result;
            const double inv = 1.0 / s;
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                result.data[c] = data[c] * inv;
            return result;
        }

        // Arithmetic assignment operators
        AETHER_INLINE Vec& operator+=(const Vec& o) {
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                data[c] += o.data[c];
            return *this;
        }

        AETHER_INLINE Vec& operator-=(const Vec& o) {
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                data[c] -= o.data[c];
            return *this;
        }

        AETHER_INLINE Vec& operator*=(const Vec& o) {
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                data[c] *= o.data[c];
            return *this;
        }

        AETHER_INLINE Vec& operator/=(const Vec& o) {
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                data[c] /= o.data[c];
            return *this;
        }

        AETHER_INLINE Vec& operator*=(double s) {
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                data[c] *= s;
            return *this;
        }

        AETHER_INLINE Vec& operator/=(double s) {
            const double inv = 1.0 / s;
            #pragma omp simd
            for (int c = 0; c < Size; ++c)
                data[c] *= inv;
            return *this;
        }

        // Math Operations
        AETHER_INLINE double dot(const Vec& o) const {
            double res = 0.0;
            #pragma omp simd reduction(+:res)
            for (int c = 0; c < Size; ++c)
                res += data[c] * o.data[c];
            return res;
        }

        AETHER_INLINE double norm2() const {
            double res = 0.0;
            #pragma omp simd reduction(+:res)
            for (int c = 0; c < Size; ++c)
                res += data[c] * data[c];
            return res;
        }

        AETHER_INLINE double norm() const {
            return std::sqrt(norm2());
        }
    };

    // scalar * Vec This is the left hand side operation overload
    // for double times vector. This is a c++ oddity lol
    template<int Size>
    AETHER_INLINE Vec<Size> operator*(double s, const Vec<Size>& v) {
        return v * s;
    }

    // ======================================================
    // Matrix temp
    // ======================================================
    template<int Size>
    struct Mat {

        std::array<Vec<Size>, Size> row{}; // row-major: row[i][j]

        Mat() = default;

        // Construct from flat array (row-major: val[i*Size + j])
        explicit Mat(const double (&val)[Size * Size]) {
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] = val[i * Size + j];
                }
            }
        }

        // Construct from flat array of Vec
        explicit Mat(const Vec<Size> (&val)[Size]) {
            for (int i = 0; i < Size; ++i) {
                row[i] = val[i];
            }
        }


        // Element access
        AETHER_INLINE double& operator()(int i, int j){ return row[i].data[j]; }
        AETHER_INLINE const double& operator()(int i, int j) const { return row[i].data[j]; }

        AETHER_INLINE Vec<Size>& operator[](int i){ return row[i]; }
        AETHER_INLINE const Vec<Size>& operator[](int i) const { return row[i]; }

        // Elementwise matrix arithmetic
        AETHER_INLINE Mat operator+(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    res(i,j) = row[i].data[j] + o(i,j);
                }
            }
            return res;
        }

        AETHER_INLINE Mat operator-(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    res(i,j) = row[i].data[j] - o(i,j);
                }
            }
            return res;
        }

        AETHER_INLINE Mat operator*(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    res(i,j) = row[i].data[j] * o(i,j);
                }
            }
            return res;
        }

        AETHER_INLINE Mat operator/(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    res(i,j) = row[i].data[j] / o(i,j);
                }
            }
            return res;
        }

        AETHER_INLINE Mat& operator+=(const Mat& o) {
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] += o(i,j);
                }
            }
            return *this;
        }

        AETHER_INLINE Mat& operator-=(const Mat& o) {
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] -= o(i,j);
                }
            }
            return *this;
        }

        // Scalar ops
        AETHER_INLINE Mat operator*(double s) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    res(i,j) = row[i].data[j] * s;
                }
            }
            return res;
        }

        AETHER_INLINE Mat operator/(double s) const {
            const double inv = 1.0 / s;
            return (*this) * inv;
        }

        AETHER_INLINE Mat& operator*=(double s) {
            for (int i = 0; i < Size; ++i) {
                #pragma omp simd
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] *= s;
                }
            }
            return *this;
        }

        AETHER_INLINE Mat& operator/=(double s) {
            const double inv = 1.0 / s;
            return (*this) *= inv;
        }

        // Matrix-vector product: y = A x
        AETHER_INLINE Vec<Size> dot(const Vec<Size>& V) const {
            Vec<Size> res;
            const double * AETHER_RESTRICT v = V.data.data();
            for (int i = 0; i < Size; ++i) {
                const double* AETHER_RESTRICT a = row[i].data.data();
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int j = 0; j < Size; ++j) {
                    sum += a[j] * v[j];
                }
                res.data[i] = sum;
            }
            return res;
        }

        // Convenience: use operator* for Mat-Vec as well
        AETHER_INLINE Vec<Size> operator*(const Vec<Size>& v) const {
            return dot(v);
        }
    };

    // scalar * Mat. Same LHS overload. Weird by neccessary apparently
    template<int Size>
    AETHER_INLINE Mat<Size> operator*(double s, const Mat<Size>& M) {
        return M * s;
    }

} // namespace aether::math
