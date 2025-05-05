#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include <array>
#include <iostream>

// Template Vector class to handle both 2D and 3D vectors
template <int D>
class Vector {
public:
    std::array<double, D> components;

    Vector() {
        for (int i = 0; i < D; i++) components[i] = 0.0;
    }

    Vector(const std::array<double, D>& vals) : components(vals) {}

    // Access components with [] operator
    double& operator[](int i) { return components[i]; }
    const double& operator[](int i) const { return components[i]; }

    // Vector addition
    Vector operator+(const Vector& other) const {
        Vector result;
        for (int i = 0; i < D; i++) result[i] = components[i] + other[i];
        return result;
    }

    // Vector subtraction
    Vector operator-(const Vector& other) const {
        Vector result;
        for (int i = 0; i < D; i++) result[i] = components[i] - other[i];
        return result;
    }

    // Vector-scalar multiplication
    Vector operator*(double scalar) const {
        Vector result;
        for (int i = 0; i < D; i++) result[i] = components[i] * scalar;
        return result;
    }

    // Vector-scalar division
    Vector operator/(double scalar) const {
        Vector result;
        for (int i = 0; i < D; i++) result[i] = components[i] / scalar;
        return result;
    }

    // Compound assignment operators
    Vector& operator+=(const Vector& other) {
        for (int i = 0; i < D; i++) components[i] += other[i];
        return *this;
    }

    Vector& operator-=(const Vector& other) {
        for (int i = 0; i < D; i++) components[i] -= other[i];
        return *this;
    }

    Vector& operator*=(double scalar) {
        for (int i = 0; i < D; i++) components[i] *= scalar;
        return *this;
    }

    Vector& operator/=(double scalar) {
        for (int i = 0; i < D; i++) components[i] /= scalar;
        return *this;
    }

    // Dot product
    double dot(const Vector& other) const {
        double sum = 0.0;
        for (int i = 0; i < D; i++) sum += components[i] * other[i];
        return sum;
    }

    // Squared magnitude
    double magnitude_squared() const {
        double sum = 0.0;
        for (int i = 0; i < D; i++) sum += components[i] * components[i];
        return sum;
    }

    // Magnitude
    double magnitude() const {
        return std::sqrt(magnitude_squared());
    }

    // Normalize
    Vector normalized() const {
        double mag = magnitude();
        if (mag < 1e-10) return Vector();
        return *this / mag;
    }

    // Distance squared between two vectors
    static double distance_squared(const Vector& a, const Vector& b) {
        Vector diff = a - b;
        return diff.magnitude_squared();
    }

    // Distance between two vectors
    static double distance(const Vector& a, const Vector& b) {
        return std::sqrt(distance_squared(a, b));
    }
};

// Add operator!= for Vector comparison
template <int D>
bool operator!=(const Vector<D>& lhs, const Vector<D>& rhs) {
    for (int i = 0; i < D; ++i) {
        if (lhs[i] != rhs[i]) {
            return true;
        }
    }
    return false;
}

// Helper functions for 2D and 3D vectors
template<>
inline Vector<3> Vector<3>::normalized() const {
    double mag = magnitude();
    if (mag < 1e-10) return Vector<3>();
    return *this / mag;
}

// Cross product for 3D vectors
inline Vector<3> cross(const Vector<3>& a, const Vector<3>& b) {
    return Vector<3>({
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    });
}

// Scalar-vector multiplication
template <int D>
inline Vector<D> operator*(double scalar, const Vector<D>& vec) {
    return vec * scalar;
}

// Type aliases for convenience
using Vector2D = Vector<2>;
using Vector3D = Vector<3>;

// Function to compute the Hilbert key for 2D and 3D points (needed for BVH)
template <int D>
unsigned long long hilbert_key(const Vector<D>& p, double min_coord, double max_coord, int bits) {
    // Convert point coordinates to integer coordinates in range [0, 2^bits-1]
    std::array<unsigned int, D> coords;
    double range = max_coord - min_coord;
    
    for (int i = 0; i < D; i++) {
        double normalized = (p[i] - min_coord) / range;
        coords[i] = static_cast<unsigned int>(normalized * ((1ULL << bits) - 1));
    }
    
    // Hilbert curve encoding
    unsigned long long key = 0;
    
    // 2D Hilbert curve
    if (D == 2) {
        for (int i = bits - 1; i >= 0; i--) {
            unsigned int bits_i = ((coords[0] >> i) & 1) | (((coords[1] >> i) & 1) << 1);
            key = (key << 2) | bits_i;
            
            // XOR operation based on highest bit
            if ((key & 0x4) != 0) {
                key ^= 0x3;  // Rotate pattern for next iteration
            }
        }
    }
    // 3D Hilbert curve
    else if (D == 3) {
        for (int i = bits - 1; i >= 0; i--) {
            unsigned int bits_i = ((coords[0] >> i) & 1) | 
                                 (((coords[1] >> i) & 1) << 1) | 
                                 (((coords[2] >> i) & 1) << 2);
            key = (key << 3) | bits_i;
            
            // More complex 3D rotation patterns
            if ((key & 0x8) != 0) {
                key ^= 0x7;
            }
        }
    }
    
    return key;
}

#endif // VECTOR_H
