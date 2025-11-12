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

    struct Mat3{
        Vec3 row1, row2, row3;
        Mat3() = default;
        Mat3(Vec3 one, Vec3 two, Vec3 three){
            row1 = Vec3(one.x,one.y,one.z); 
            row2 = Vec3(two.x,two.y,two.z);
            row3 = Vec3(three.x,three.y,three.z);
        }
        Mat3(double one, double two, double three,
             double four, double five, double six,
             double seven, double eight, double nine){
            row1 = Vec3(one,two,three);
            row2 = Vec3(four, five, six);
            row3 = Vec3(seven,eight,nine);
        }

        AETHER_INLINE Mat3 operator+(const Mat3 &o) const{
            return {row1 + o.row1, row2 + o.row2, row3 + o.row3};
        }
        AETHER_INLINE Mat3 operator-(const Mat3 &o) const{
            return {row1 - o.row1, row2 - o.row2, row3 - o.row3};
        }
        AETHER_INLINE Mat3 operator*(const Mat3 &o) const{
            return {row1 * o.row1, row2 * o.row2, row3 * o.row3};
        }
        AETHER_INLINE Mat3 operator/(const Mat3 &o) const{
            return {row1 / o.row1, row2 / o.row2, row3 / o.row3};
        }

        AETHER_INLINE Vec3 dot(const Vec3 o){
            return {row1.dot(o), row2.dot(o),row3.dot(o)};
        }

    };
}