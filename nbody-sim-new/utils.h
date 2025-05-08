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
#include <parlay/parallel.h>  // For parlay::parallel_for
#include <parlay/primitives.h>  // For parlay::reduce

// Physical constants
const double G = 4.471e-21; // For distances in AU, masses in Earth masses  // Gravitational constant G= 6.67430e-11 m^3 kg^-1 s^-2
const double BARNES_HUT_THETA = 0.25;  // Barnes-Hut approximation parameter
const double EPSILON = 1e-11;  // Small value to avoid division by zero
const double SOFTENING = 1e-6;  // Softening parameter for close interactions
const double ACCURACY_PCT_THRESHOLD = 0.01;  // Threshold % for accuracy: if value is within 1% it is accurate
const double ACCURACY_FORCE_THRESHOLD = 1e-20;  // Force threshold for accuracy checks
const double MASS_THRESHOLD = 1e-10;

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
        << "_"
        << std::setw(2) << tm_now->tm_hour          // hour
        << std::setw(2) << tm_now->tm_min           // minute
        << std::setw(2) << tm_now->tm_sec;          // second
    
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
    std::uniform_real_distribution<double> position_dist(1, 10000000.0);
    std::uniform_real_distribution<double> velocity_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> mass_dist(1, 100000000.0);
    
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

// Optimized OpenMP version of accuracy computation for large datasets
template <int D>
inline double compute_accuracy_omp(const std::vector<Vector<D>>& forces, 
                               const std::vector<Vector<D>>& reference_forces) {
    if (forces.size() != reference_forces.size()) {
        std::cerr << "Error: Force vector sizes do not match for accuracy calculation." << std::endl;
        return 0.0;
    }
    
    const size_t n = forces.size();
    std::vector<int> accurate_flags(n, 0);
    const double accuracy_threshold = ACCURACY_PCT_THRESHOLD; // 1% accuracy threshold
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        bool is_accurate = true;
        
        for (int d = 0; d < D; ++d) {
            double ref_force = reference_forces[i][d];
            double force = forces[i][d];
            
            // Skip near-zero forces to avoid division by small numbers
            if (std::abs(ref_force) < ACCURACY_FORCE_THRESHOLD) {
                // For very small forces, check absolute error instead
                if (std::abs(force) > 1e-9) {
                    is_accurate = false;
                    break;
                }
                continue;
            }
            
            double relative_error = std::abs((force - ref_force) / ref_force);
            if (relative_error > accuracy_threshold) {
                is_accurate = false;
                break;
            }
        }
        
        if (is_accurate) {
            accurate_flags[i] = 1;
        }
    }
    
    // Count accurate bodies
    int accurate_count = 0;
    for (size_t i = 0; i < n; ++i) {
        accurate_count += accurate_flags[i];
    }
    
    return 100.0 * accurate_count / n; // Return percentage
}

// Optimized ParlayLib version of accuracy computation for large datasets
template <int D>
inline double compute_accuracy_parlay_opt(const parlay::sequence<Vector<D>>& forces, 
                                     const parlay::sequence<Vector<D>>& reference_forces) {
    if (forces.size() != reference_forces.size()) {
        std::cerr << "Error: Force vector sizes do not match for accuracy calculation." << std::endl;
        return 0.0;
    }
    
    const size_t n = forces.size();
    const double accuracy_threshold = ACCURACY_PCT_THRESHOLD; // 1% accuracy threshold
    
    // Create a sequence to store accuracy flags using parallel_for
    parlay::sequence<int> accurate_flags(n);
    
    parlay::parallel_for(0, n, [&](size_t i) {
        bool is_accurate = true;
        
        for (int d = 0; d < D; ++d) {
            double ref_force = reference_forces[i][d];
            double force = forces[i][d];
            
            // Skip near-zero forces to avoid division by small numbers
            if (std::abs(ref_force) < ACCURACY_FORCE_THRESHOLD) {
                // For very small forces, check absolute error instead
                if (std::abs(force) > 1e-9) {
                    is_accurate = false;
                    break;
                }
                continue;
            }
            
            double relative_error = std::abs((force - ref_force) / ref_force);
            if (relative_error > accuracy_threshold) {
                is_accurate = false;
                break;
            }
        }
        
        accurate_flags[i] = is_accurate ? 1 : 0;
    });
    
    // Count accurate bodies using reduce
    int accurate_count = parlay::reduce(accurate_flags, parlay::plus<int>());
    
    return 100.0 * accurate_count / n; // Return percentage
}

// For backward compatibility, define compute_accuracy
template <int D>
inline double compute_accuracy(const std::vector<Vector<D>>& forces, 
                           const std::vector<Vector<D>>& reference_forces) {
    return compute_accuracy_omp<D>(forces, reference_forces);
}

#endif // UTILS_H
