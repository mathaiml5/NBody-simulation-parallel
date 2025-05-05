#ifndef UTILS_H
#define UTILS_H

#include <filesystem>
#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include "vector.h"
#include "body.h"
#include <parlay/sequence.h>

// Physical constants
const double G = 6.67430e-11;  // Gravitational constant
const double BARNES_HUT_THETA = 0.25;  // Barnes-Hut approximation parameter

// Complex number utilities
inline std::complex<double> to_complex(const Vector<2>& v) {
    return std::complex<double>(v[0], v[1]);
}

inline Vector<2> from_complex(const std::complex<double>& z) {
    std::array<double, 2> values = {z.real(), z.imag()};
    return Vector<2>(values);
}

// Mathematical utility functions
inline int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Helper method for binomial coefficient calculation
inline int binomial(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    
    int result = 1;
    for (int i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// Create results directory if it doesn't exist
inline void ensure_results_directory() {
    std::filesystem::path results_dir = "results";
    if (!std::filesystem::exists(results_dir)) {
        std::filesystem::create_directories(results_dir);
    }
}

// Get current date/time for run ID
inline std::string get_run_id() {
    auto now = std::chrono::system_clock::now();
    std::time_t time_now = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_now = std::localtime(&time_now);
    
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << (tm_now->tm_mon + 1)     // month (0-based)
        << std::setw(2) << tm_now->tm_mday          // day
        << std::setw(4) << (tm_now->tm_year + 1900) // year
        << std::setw(2) << tm_now->tm_hour          // hour
        << std::setw(2) << tm_now->tm_min;          // minute
    
    return oss.str();
}

// Function to safely run a method and return execution time
// Returns -1 if the method fails
template<typename Func, typename... Args>
inline long long safely_execute(std::ofstream& log_file, const std::string& method_name, Func&& func, Args&&... args) {
    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = std::forward<Func>(func)(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return duration;
    } catch (const std::exception& e) {
        log_file << "Error executing " << method_name << ": " << e.what() << std::endl;
        std::cerr << "Error executing " << method_name << ": " << e.what() << std::endl;
        return -1;
    } catch (...) {
        log_file << "Unknown error executing " << method_name << std::endl;
        std::cerr << "Unknown error executing " << method_name << std::endl;
        return -1;
    }
}

// Function to generate random bodies with uniform distribution
template <int D>
inline std::vector<Body<D>> generate_random_bodies(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Define distributions with the specified ranges
    std::uniform_real_distribution<double> position_dist(0.0, 1000000.0);
    std::uniform_real_distribution<double> velocity_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> mass_dist(10.0, 10000.0);
    
    // Reserve space for efficiency
    std::vector<Body<D>> bodies;
    bodies.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        Vector<D> position;
        Vector<D> velocity;
        
        for (int d = 0; d < D; ++d) {
            position[d] = position_dist(gen);
            velocity[d] = velocity_dist(gen);
        }
        
        double mass = mass_dist(gen);
        bodies.emplace_back(position, velocity, mass);
    }
    
    return bodies;
}

// Helper function to print force values for validation (std::vector version)
template <int D>
inline void print_validation_forces(const std::vector<Vector<D>>& forces, int n, std::ostream& out) {
    int print_total = 3;
    for (int i = 0; i < n; i++) {
        if ((i + 1) % (n / print_total) == 0) {
            out << "Body #" << i + 1 << " force: (";
            for (int d = 0; d < D; d++) {
                out << forces[i][d];
                if (d < D - 1) out << ", ";
            }
            out << ")" << std::endl;
        }
    }
}

// Helper function to print parlay forces for validation
template <int D>
inline void print_validation_forces(const parlay::sequence<Vector<D>>& forces, int n, std::ostream& out) {
    int print_total = 3;
    for (int i = 0; i < n; i++) {
        if ((i + 1) % (n / print_total) == 0) {
            out << "Body #" << i + 1 << " force: (";
            for (int d = 0; d < D; d++) {
                out << forces[i][d];
                if (d < D - 1) out << ", ";
            }
            out << ")" << std::endl;
        }
    }
}

#endif // UTILS_H
