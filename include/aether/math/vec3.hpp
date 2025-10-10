#pragma once
#include <cmath>
#include <aether/core/config.hpp> 



namespace aether::math{
    struct Vec3{
        double x{0.0},y{0.0},z{0.0};
        AETHER_HD Vec3() = default;
        AETHER_HD Vec3(double X, double Y, double Z) : x(X), y(Y), z(Z){}

        // Arithmetic Operators 
        AETHER_HD AETHER_INLINE Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z};}
        AETHER_HD AETHER_INLINE Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z};}
        AETHER_HD AETHER_INLINE Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z};}
        AETHER_HD AETHER_INLINE Vec3 operator/(const Vec3& o) const { return {x / o.x, y / o.y, z / o.z};}

        // Arithmetic Assinment Operators 
        AETHER_HD AETHER_INLINE Vec3 operator+=(const Vec3& o) { return {x += o.x, y += o.y, z += o.z};}
        AETHER_HD AETHER_INLINE Vec3 operator-=(const Vec3& o) { return {x -= o.x, y -= o.y, z -= o.z};}
        AETHER_HD AETHER_INLINE Vec3 operator*=(const Vec3& o) { return {x *= o.x, y *= o.y, z *= o.z};}
        AETHER_HD AETHER_INLINE Vec3 operator/=(const Vec3& o) { return {x /= o.x, y /= o.y, z /= o.z};}

        // Math Operations
        AETHER_HD AETHER_INLINE double dot(const Vec3& o) const { return (x * o.x) + (y * o.y) + (z * o.z);}
        AETHER_HD AETHER_INLINE double norm2() const { return (x * x) + (y * y) + (z * z);}
        AETHER_HD AETHER_INLINE double norm() const { return std::sqrt(norm2());}
    };
}