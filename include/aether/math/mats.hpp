#pragma once
#include <array>
#include <cmath>
#include <Kokkos_Macros.hpp>
#include <aether/core/config.hpp>

namespace aether::math {

    // ======================================================
    // Vector temp
    // ======================================================
    template<int Size>
    struct Vec {
        std::array<double, Size> data{};
        static constexpr int size = Size;

        KOKKOS_INLINE_FUNCTION
        Vec() = default;

        // Construct from array
        KOKKOS_INLINE_FUNCTION
        explicit Vec(const double (&vals)[Size]) {
            for (int c = 0; c < Size; ++c) {
                data[c] = vals[c];
            }
        }

        // Element access
        KOKKOS_INLINE_FUNCTION
        double& operator[](int i) { return data[i]; }

        KOKKOS_INLINE_FUNCTION
        const double& operator[](int i) const { return data[i]; }

        // Arithmetic Operators
        KOKKOS_INLINE_FUNCTION
        Vec operator+(const Vec& o) const {
            Vec result;
            for (int c = 0; c < Size; ++c) {
                result.data[c] = data[c] + o.data[c];
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION
        Vec operator-(const Vec& o) const {
            Vec result;
            for (int c = 0; c < Size; ++c) {
                result.data[c] = data[c] - o.data[c];
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION
        Vec operator*(const Vec& o) const {
            Vec result;
            for (int c = 0; c < Size; ++c) {
                result.data[c] = data[c] * o.data[c];
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION
        Vec operator/(const Vec& o) const {
            Vec result;
            for (int c = 0; c < Size; ++c) {
                result.data[c] = data[c] / o.data[c];
            }
            return result;
        }

        // Scalar ops
        KOKKOS_INLINE_FUNCTION
        Vec operator*(double s) const {
            Vec result;
            for (int c = 0; c < Size; ++c) {
                result.data[c] = data[c] * s;
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION
        Vec operator/(double s) const {
            Vec result;
            const double inv = 1.0 / s;
            for (int c = 0; c < Size; ++c) {
                result.data[c] = data[c] * inv;
            }
            return result;
        }

        // Arithmetic assignment operators
        KOKKOS_INLINE_FUNCTION
        Vec& operator+=(const Vec& o) {
            for (int c = 0; c < Size; ++c) {
                data[c] += o.data[c];
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Vec& operator-=(const Vec& o) {
            for (int c = 0; c < Size; ++c) {
                data[c] -= o.data[c];
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Vec& operator*=(const Vec& o) {
            for (int c = 0; c < Size; ++c) {
                data[c] *= o.data[c];
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Vec& operator/=(const Vec& o) {
            for (int c = 0; c < Size; ++c) {
                data[c] /= o.data[c];
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Vec& operator*=(double s) {
            for (int c = 0; c < Size; ++c) {
                data[c] *= s;
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Vec& operator/=(double s) {
            const double inv = 1.0 / s;
            for (int c = 0; c < Size; ++c) {
                data[c] *= inv;
            }
            return *this;
        }

        // Math Operations
        KOKKOS_INLINE_FUNCTION
        double dot(const Vec& o) const {
            double res = 0.0;
            for (int c = 0; c < Size; ++c) {
                res += data[c] * o.data[c];
            }
            return res;
        }

        KOKKOS_INLINE_FUNCTION
        double norm2() const {
            double res = 0.0;
            for (int c = 0; c < Size; ++c) {
                res += data[c] * data[c];
            }
            return res;
        }

        KOKKOS_INLINE_FUNCTION
        double norm() const {
            return sqrt(norm2());
        }
    };

    // scalar * Vec
    template<int Size>
    KOKKOS_INLINE_FUNCTION
    Vec<Size> operator*(double s, const Vec<Size>& v) {
        return v * s;
    }

    // ======================================================
    // Matrix temp
    // ======================================================
    template<int Size>
    struct Mat {
        std::array<Vec<Size>, Size> row{}; // row-major: row[i][j]

        KOKKOS_INLINE_FUNCTION
        Mat() = default;

        // Construct from flat array (row-major: val[i*Size + j])
        KOKKOS_INLINE_FUNCTION
        explicit Mat(const double (&val)[Size * Size]) {
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] = val[i * Size + j];
                }
            }
        }

        // Construct from array of Vec
        KOKKOS_INLINE_FUNCTION
        explicit Mat(const Vec<Size> (&val)[Size]) {
            for (int i = 0; i < Size; ++i) {
                row[i] = val[i];
            }
        }

        // Element access
        KOKKOS_INLINE_FUNCTION
        double& operator()(int i, int j) { return row[i].data[j]; }

        KOKKOS_INLINE_FUNCTION
        const double& operator()(int i, int j) const { return row[i].data[j]; }

        KOKKOS_INLINE_FUNCTION
        Vec<Size>& operator[](int i) { return row[i]; }

        KOKKOS_INLINE_FUNCTION
        const Vec<Size>& operator[](int i) const { return row[i]; }

        // Elementwise matrix arithmetic
        KOKKOS_INLINE_FUNCTION
        Mat operator+(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    res(i, j) = row[i].data[j] + o(i, j);
                }
            }
            return res;
        }

        KOKKOS_INLINE_FUNCTION
        Mat operator-(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    res(i, j) = row[i].data[j] - o(i, j);
                }
            }
            return res;
        }

        KOKKOS_INLINE_FUNCTION
        Mat operator*(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    res(i, j) = row[i].data[j] * o(i, j);
                }
            }
            return res;
        }

        KOKKOS_INLINE_FUNCTION
        Mat operator/(const Mat& o) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    res(i, j) = row[i].data[j] / o(i, j);
                }
            }
            return res;
        }

        KOKKOS_INLINE_FUNCTION
        Mat& operator+=(const Mat& o) {
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] += o(i, j);
                }
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Mat& operator-=(const Mat& o) {
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] -= o(i, j);
                }
            }
            return *this;
        }

        // Scalar ops
        KOKKOS_INLINE_FUNCTION
        Mat operator*(double s) const {
            Mat res;
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    res(i, j) = row[i].data[j] * s;
                }
            }
            return res;
        }

        KOKKOS_INLINE_FUNCTION
        Mat operator/(double s) const {
            const double inv = 1.0 / s;
            return (*this) * inv;
        }

        KOKKOS_INLINE_FUNCTION
        Mat& operator*=(double s) {
            for (int i = 0; i < Size; ++i) {
                for (int j = 0; j < Size; ++j) {
                    row[i].data[j] *= s;
                }
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Mat& operator/=(double s) {
            const double inv = 1.0 / s;
            return (*this) *= inv;
        }

        // Matrix-vector product: y = A x
        KOKKOS_INLINE_FUNCTION
        Vec<Size> dot(const Vec<Size>& V) const {
            Vec<Size> res;
            for (int i = 0; i < Size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < Size; ++j) {
                    sum += row[i].data[j] * V.data[j];
                }
                res.data[i] = sum;
            }
            return res;
        }

        // Convenience: use operator* for Mat-Vec as well
        KOKKOS_INLINE_FUNCTION
        Vec<Size> operator*(const Vec<Size>& v) const {
            return dot(v);
        }
    };

    // scalar * Mat
    template<int Size>
    KOKKOS_INLINE_FUNCTION
    Mat<Size> operator*(double s, const Mat<Size>& M) {
        return M * s;
    }

    // Helpful aliases if you want them
    template<int Size>
    using Max = Mat<Size>;

} // namespace aether::math