#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iomanip>
#include <signal.h>
#include <filesystem>
#include <omp.h>

#include "vector.h"
#include "body.h"
#include "utils.h"  // Include utils.h before methods.h (now contains accuracy functions)
#include "methods.h"
#include <iomanip>  // For std::setprecision

// Helper for running and timing different n-body methods
template <int D>
void run_benchmark(const std::vector<Body<D>>& bodies, const std::string& run_id, bool calculate_accuracy, 
                  const std::string& methods = "abfh", bool override_bf_limit = false) {
    int n = bodies.size();
    
    // Check if brute force should be skipped for large datasets
    bool run_bruteforce = methods.find('a') != std::string::npos && (n <= 1000000 || override_bf_limit);
    bool run_barneshut = methods.find('b') != std::string::npos;
    bool run_bvh = methods.find('h') != std::string::npos;
    bool run_fmm = methods.find('f') != std::string::npos;
    
    // If no methods specified, run all
    if (methods.empty()) {
        run_bruteforce = n <= 1000000; // Only run brute force for N <= 1M
        run_barneshut = true;
        run_bvh = true;
        run_fmm = true;
    }
    
    // Ensure results directory exists
    ensure_results_directory();
    
    // Create output files with proper naming convention in results subfolder
    std::string base_name = "run_" + run_id + "_N_" + std::to_string(n) + "_" + std::to_string(D) + "D";
    std::string csv_file = "results/" + base_name + ".csv";
    std::string out_file = "results/" + base_name + ".out";
    
    std::ofstream csv_output(csv_file);
    if (!csv_output.is_open()) {
        std::cerr << "Failed to open output file: " << csv_file << std::endl;
        return;
    }
    
    std::ofstream log_output(out_file);
    if (!log_output.is_open()) {
        std::cerr << "Failed to open log file: " << out_file << std::endl;
        csv_output.close();
        return;
    }
    
    // Write headers to CSV - new format with dimension and conditional accuracy column
    csv_output << "Method,Bodies,Dimension,Time(s)";
    if (calculate_accuracy) {
        csv_output << ",Accuracy(%)";
    }
    csv_output << std::endl;
    
    // Write benchmark information to log
    log_output << "Running N-body simulation benchmark with:" << std::endl;
    log_output << "  Dimension: " << D << "D" << std::endl;
    log_output << "  Bodies: " << n << std::endl;
    log_output << "  Run ID: " << run_id << std::endl;
    if (calculate_accuracy) {
        log_output << "  Accuracy calculation: ON" << std::endl;
    }
    log_output << "  Methods: ";
    if (run_bruteforce) log_output << "Brute Force ";
    if (run_barneshut) log_output << "Barnes-Hut ";
    if (run_bvh) log_output << "BVH ";
    if (run_fmm) log_output << "FMM ";
    log_output << std::endl << std::endl;
    
    std::cout << "Running N-body simulation benchmark with:" << std::endl;
    std::cout << "  Dimension: " << D << "D" << std::endl;
    std::cout << "  Bodies: " << n << std::endl;
    std::cout << "  Run ID: " << run_id << std::endl;
    if (calculate_accuracy) {
        std::cout << "  Accuracy calculation: ON" << std::endl;
    }
    std::cout << "  Methods: ";
    if (run_bruteforce) std::cout << "Brute Force ";
    if (run_barneshut) std::cout << "Barnes-Hut ";
    if (run_bvh) std::cout << "BVH ";
    if (run_fmm) std::cout << "FMM ";
    std::cout << std::endl << std::endl;

    // Convert to parlay sequence for reference calculation and parlay methods
    parlay::sequence<Body<D>> parlay_bodies(bodies.begin(), bodies.end());

    // Calculate reference solution only if accuracy calculation is enabled
    std::vector<Vector<D>> reference_forces;
    parlay::sequence<Vector<D>> parlay_reference_forces;
    bool using_parlay_reference = false;

    if (calculate_accuracy) {
        if (n < 100000) {
            // For smaller datasets, use brute force sequential as reference
            std::cout << "Using brute force sequential as reference for accuracy calculation..." << std::endl;
            log_output << "Using brute force sequential as reference for accuracy calculation..." << std::endl;
            reference_forces = brute_force_seq_n_body<D>(bodies);
        } else {
            // For larger datasets, use parallel implementations as reference
            if (n < 50000000) {
                // Use OpenMP for medium-sized datasets
                std::cout << "Using brute force OpenMP as reference for accuracy calculation..." << std::endl;
                log_output << "Using brute force OpenMP as reference for accuracy calculation..." << std::endl;
                reference_forces = brute_force_omp_n_body_2<D>(bodies);
            } else {
                // Use Parlay for very large datasets
                std::cout << "Using brute force Parlay as reference for accuracy calculation..." << std::endl;
                log_output << "Using brute force Parlay as reference for accuracy calculation..." << std::endl;
                using_parlay_reference = true;
                parlay_reference_forces = brute_force_parlay_n_body_2<D>(parlay_bodies);
                // Also store in std::vector format for methods that need it
                reference_forces = std::vector<Vector<D>>(parlay_reference_forces.begin(), parlay_reference_forces.end());
            }
        }
    } else {
        std::cout << "Accuracy calculation disabled." << std::endl;
        log_output << "Accuracy calculation disabled." << std::endl;
    }

    // Brute force methods - only run if specified
    if (run_bruteforce) {
        // Brute force sequential
        log_output << "Brute force O(n²) sequential approach:" << std::endl;
        std::cout << "Brute force O(n²) sequential approach:" << std::endl;
        
        std::vector<Vector<D>> forces_bf_seq;
        auto time_bf_seq = safely_execute(log_output, "BruteForce_Sequential", [&]() {
            forces_bf_seq = brute_force_seq_n_body<D>(bodies);
            return forces_bf_seq;
        });
        
        // Calculate accuracy separately, outside of the timed execution
        double accuracy_bf_seq = -1.0;
        if (calculate_accuracy) {
            if (using_parlay_reference) {
                // If using Parlay reference, convert to std::vector temporarily for comparison
                accuracy_bf_seq = compute_accuracy_omp(forces_bf_seq, reference_forces);
            } else {
                // Always 100% accurate compared to itself if it's the reference
                accuracy_bf_seq = n < 10000 ? 100.0 : compute_accuracy_omp(forces_bf_seq, reference_forces);
            }
        }
        
        // Convert time from microseconds to seconds
        double time_bf_seq_seconds = time_bf_seq / 1e6;
        
        if (time_bf_seq >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BruteForce_Sequential," << n << "," << D;
            if (time_bf_seq_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bf_seq_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bf_seq_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bf_seq;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bf_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bf_seq >= 0 ? std::to_string(accuracy_bf_seq) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bf_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bf_seq >= 0 ? std::to_string(accuracy_bf_seq) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_bf_seq, n, log_output);
            print_validation_forces(forces_bf_seq, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // Brute force OpenMP (method 1)
        log_output << "Brute force OpenMP parallel approach (memory-intensive):" << std::endl;
        log_output << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        std::cout << "Brute force OpenMP parallel approach (memory-intensive):" << std::endl;
        std::cout << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        
        std::vector<Vector<D>> forces_bf_omp1;
        auto time_bf_omp1 = safely_execute(log_output, "BruteForce_OpenMP1", [&]() {
            forces_bf_omp1 = brute_force_omp_n_body_1<D>(bodies);
            return forces_bf_omp1;
        });
        
        double accuracy_bf_omp1 = -1.0;
        if (calculate_accuracy) {
            accuracy_bf_omp1 = compute_accuracy_omp(forces_bf_omp1, reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_bf_omp1_seconds = time_bf_omp1 / 1e6;
        
        if (time_bf_omp1 >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BruteForce_OpenMP1," << n << "," << D;
            if (time_bf_omp1_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bf_omp1_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bf_omp1_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bf_omp1;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bf_omp1_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bf_omp1 >= 0 ? std::to_string(accuracy_bf_omp1) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bf_omp1_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bf_omp1 >= 0 ? std::to_string(accuracy_bf_omp1) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_bf_omp1, n, log_output);
            print_validation_forces(forces_bf_omp1, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // Brute force OpenMP (method 2)
        log_output << "Brute force OpenMP parallel approach (memory-efficient):" << std::endl;
        log_output << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        std::cout << "Brute force OpenMP parallel approach (memory-efficient):" << std::endl;
        std::cout << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        
        std::vector<Vector<D>> forces_bf_omp2;
        auto time_bf_omp2 = safely_execute(log_output, "BruteForce_OpenMP2", [&]() {
            forces_bf_omp2 = brute_force_omp_n_body_2<D>(bodies);
            return forces_bf_omp2;
        });
        
        double accuracy_bf_omp2 = -1.0;
        if (calculate_accuracy) {
            if (using_parlay_reference) {
                accuracy_bf_omp2 = compute_accuracy_omp(forces_bf_omp2, reference_forces);
            } else if (n >= 10000 && n < 50000) {
                // This is the reference for medium datasets
                accuracy_bf_omp2 = 100.0;
            } else {
                accuracy_bf_omp2 = compute_accuracy_omp(forces_bf_omp2, reference_forces);
            }
        }
        
        // Convert time from microseconds to seconds
        double time_bf_omp2_seconds = time_bf_omp2 / 1e6;
        
        if (time_bf_omp2 >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BruteForce_OpenMP2," << n << "," << D;
            if (time_bf_omp2_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bf_omp2_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bf_omp2_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bf_omp2;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bf_omp2_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bf_omp2 >= 0 ? std::to_string(accuracy_bf_omp2) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bf_omp2_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bf_omp2 >= 0 ? std::to_string(accuracy_bf_omp2) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_bf_omp2, n, log_output);
            print_validation_forces(forces_bf_omp2, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // Brute force Parlay (method 1)
        log_output << "Brute force ParlayLib parallel approach (memory-intensive):" << std::endl;
        log_output << "Using " << parlay::num_workers() << " workers..." << std::endl;
        std::cout << "Brute force ParlayLib parallel approach (memory-intensive):" << std::endl;
        std::cout << "Using " << parlay::num_workers() << " workers..." << std::endl;
        
        parlay::sequence<Vector<D>> forces_bf_parlay1;
        auto time_bf_parlay1 = safely_execute(log_output, "BruteForce_Parlay1", [&]() {
            forces_bf_parlay1 = brute_force_parlay_n_body_1<D>(parlay_bodies);
            return forces_bf_parlay1;
        });
        
        double accuracy_bf_parlay1 = -1.0;
        if (calculate_accuracy) {
            if (using_parlay_reference) {
                // Using Parlay optimized comparison for large datasets
                accuracy_bf_parlay1 = compute_accuracy_parlay_opt(forces_bf_parlay1, parlay_reference_forces);
            } else {
                // Convert to std::vector and use OpenMP comparison
                accuracy_bf_parlay1 = compute_accuracy_omp(
                    std::vector<Vector<D>>(forces_bf_parlay1.begin(), forces_bf_parlay1.end()),
                    reference_forces
                );
            }
        }
        
        // Convert time from microseconds to seconds
        double time_bf_parlay1_seconds = time_bf_parlay1 / 1e6;
        
        if (time_bf_parlay1 >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BruteForce_Parlay1," << n << "," << D;
            if (time_bf_parlay1_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bf_parlay1_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bf_parlay1_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bf_parlay1;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bf_parlay1_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bf_parlay1 >= 0 ? std::to_string(accuracy_bf_parlay1) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bf_parlay1_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bf_parlay1 >= 0 ? std::to_string(accuracy_bf_parlay1) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(std::vector<Vector<D>>(forces_bf_parlay1.begin(), forces_bf_parlay1.end()), n, log_output);
            print_validation_forces(std::vector<Vector<D>>(forces_bf_parlay1.begin(), forces_bf_parlay1.end()), n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // Brute force Parlay (method 2)
        log_output << "Brute force ParlayLib parallel approach (memory-efficient):" << std::endl;
        log_output << "Using " << parlay::num_workers() << " workers..." << std::endl;
        std::cout << "Brute force ParlayLib parallel approach (memory-efficient):" << std::endl;
        std::cout << "Using " << parlay::num_workers() << " workers..." << std::endl;
        
        parlay::sequence<Vector<D>> forces_bf_parlay2;
        auto time_bf_parlay2 = safely_execute(log_output, "BruteForce_Parlay2", [&]() {
            forces_bf_parlay2 = brute_force_parlay_n_body_2<D>(parlay_bodies);
            return forces_bf_parlay2;
        });
        
        double accuracy_bf_parlay2 = -1.0;
        if (calculate_accuracy) {
            if (using_parlay_reference && n >= 50000) {
                // This is the reference for large datasets
                accuracy_bf_parlay2 = 100.0;
            } else if (using_parlay_reference) {
                // Using Parlay optimized comparison for large datasets
                accuracy_bf_parlay2 = compute_accuracy_parlay_opt(forces_bf_parlay2, parlay_reference_forces);
            } else {
                // Convert to std::vector and use OpenMP comparison
                accuracy_bf_parlay2 = compute_accuracy_omp(
                    std::vector<Vector<D>>(forces_bf_parlay2.begin(), forces_bf_parlay2.end()),
                    reference_forces
                );
            }
        }
        
        // Convert time from microseconds to seconds
        double time_bf_parlay2_seconds = time_bf_parlay2 / 1e6;
        
        if (time_bf_parlay2 >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BruteForce_Parlay2," << n << "," << D;
            if (time_bf_parlay2_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bf_parlay2_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bf_parlay2_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bf_parlay2;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bf_parlay2_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bf_parlay2 >= 0 ? std::to_string(accuracy_bf_parlay2) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bf_parlay2_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bf_parlay2 >= 0 ? std::to_string(accuracy_bf_parlay2) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(std::vector<Vector<D>>(forces_bf_parlay2.begin(), forces_bf_parlay2.end()), n, log_output);
            print_validation_forces(std::vector<Vector<D>>(forces_bf_parlay2.begin(), forces_bf_parlay2.end()), n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
    }
    
    // Barnes-Hut methods - only run if specified
    if (run_barneshut) {
        // Barnes-Hut sequential
        log_output << "Barnes-Hut sequential approach:" << std::endl;
        std::cout << "Barnes-Hut sequential approach:" << std::endl;
        
        std::vector<Vector<D>> forces_bh_seq;
        auto time_bh_seq = safely_execute(log_output, "BarnesHut_Sequential", [&]() {
            forces_bh_seq = barnes_hut_seq_n_body<D>(bodies);
            return forces_bh_seq;
        });
        
        double accuracy_bh_seq = -1.0;
        if (calculate_accuracy) {
            accuracy_bh_seq = compute_accuracy(forces_bh_seq, reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_bh_seq_seconds = time_bh_seq / 1e6;
        
        if (time_bh_seq >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BarnesHut_Sequential," << n << "," << D;
            if (time_bh_seq_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bh_seq_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bh_seq_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bh_seq;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bh_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bh_seq >= 0 ? std::to_string(accuracy_bh_seq) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bh_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bh_seq >= 0 ? std::to_string(accuracy_bh_seq) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_bh_seq, n, log_output);
            print_validation_forces(forces_bh_seq, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // Barnes-Hut OpenMP
        log_output << "Barnes-Hut OpenMP parallel approach:" << std::endl;
        log_output << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        std::cout << "Barnes-Hut OpenMP parallel approach:" << std::endl;
        std::cout << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        
        std::vector<Vector<D>> forces_bh_omp;
        auto time_bh_omp = safely_execute(log_output, "BarnesHut_OpenMP", [&]() {
            forces_bh_omp = barnes_hut_omp_n_body<D>(bodies);
            return forces_bh_omp;
        });
        
        double accuracy_bh_omp = -1.0;
        if (calculate_accuracy) {
            accuracy_bh_omp = compute_accuracy(forces_bh_omp, reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_bh_omp_seconds = time_bh_omp / 1e6;
        
        if (time_bh_omp >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BarnesHut_OpenMP," << n << "," << D;
            if (time_bh_omp_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bh_omp_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bh_omp_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bh_omp;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bh_omp_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bh_omp >= 0 ? std::to_string(accuracy_bh_omp) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bh_omp_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bh_omp >= 0 ? std::to_string(accuracy_bh_omp) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_bh_omp, n, log_output);
            print_validation_forces(forces_bh_omp, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;

        // Barnes-Hut Parlay
        log_output << "Barnes-Hut ParlayLib parallel approach:" << std::endl;
        log_output << "Using " << parlay::num_workers() << " workers..." << std::endl;
        std::cout << "Barnes-Hut ParlayLib parallel approach:" << std::endl;
        std::cout << "Using " << parlay::num_workers() << " workers..." << std::endl;
        
        parlay::sequence<Vector<D>> forces_bh_parlay;
        auto time_bh_parlay = safely_execute(log_output, "BarnesHut_Parlay", [&]() {
            forces_bh_parlay = barnes_hut_parlay_n_body<D>(parlay_bodies);
            return forces_bh_parlay;
        });
        
        double accuracy_bh_parlay = -1.0;
        if (calculate_accuracy) {
            accuracy_bh_parlay = compute_accuracy(std::vector<Vector<D>>(forces_bh_parlay.begin(), forces_bh_parlay.end()), reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_bh_parlay_seconds = time_bh_parlay / 1e6;
        
        if (time_bh_parlay >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BarnesHut_Parlay," << n << "," << D;
            if (time_bh_parlay_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bh_parlay_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bh_parlay_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bh_parlay;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bh_parlay_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bh_parlay >= 0 ? std::to_string(accuracy_bh_parlay) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bh_parlay_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bh_parlay >= 0 ? std::to_string(accuracy_bh_parlay) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(std::vector<Vector<D>>(forces_bh_parlay.begin(), forces_bh_parlay.end()), n, log_output);
            print_validation_forces(std::vector<Vector<D>>(forces_bh_parlay.begin(), forces_bh_parlay.end()), n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
    }

    // BVH methods - only run if specified
    if (run_bvh) {
        // BVH sequential
        log_output << "BVH sequential approach:" << std::endl;
        std::cout << "BVH sequential approach:" << std::endl;
        
        std::vector<Vector<D>> forces_bvh_seq;
        auto time_bvh_seq = safely_execute(log_output, "BVH_Sequential", [&]() {
            forces_bvh_seq = bvh_seq_n_body<D>(bodies);
            return forces_bvh_seq;
        });
        
        double accuracy_bvh_seq = -1.0;
        if (calculate_accuracy) {
            accuracy_bvh_seq = compute_accuracy(forces_bvh_seq, reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_bvh_seq_seconds = time_bvh_seq / 1e6;
        
        if (time_bvh_seq >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BVH_Sequential," << n << "," << D;
            if (time_bvh_seq_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bvh_seq_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bvh_seq_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bvh_seq;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bvh_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bvh_seq >= 0 ? std::to_string(accuracy_bvh_seq) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bvh_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bvh_seq >= 0 ? std::to_string(accuracy_bvh_seq) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_bvh_seq, n, log_output);
            print_validation_forces(forces_bvh_seq, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // BVH OpenMP
        log_output << "BVH OpenMP parallel approach:" << std::endl;
        log_output << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        std::cout << "BVH OpenMP parallel approach:" << std::endl;
        std::cout << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        
        std::vector<Vector<D>> forces_bvh_omp;
        auto time_bvh_omp = safely_execute(log_output, "BVH_OpenMP", [&]() {
            forces_bvh_omp = bvh_omp_n_body<D>(bodies);
            return forces_bvh_omp;
        });
        
        double accuracy_bvh_omp = -1.0;
        if (calculate_accuracy) {
            accuracy_bvh_omp = compute_accuracy(forces_bvh_omp, reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_bvh_omp_seconds = time_bvh_omp / 1e6;
        
        if (time_bvh_omp >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BVH_OpenMP," << n << "," << D;
            if (time_bvh_omp_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bvh_omp_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bvh_omp_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bvh_omp;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bvh_omp_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bvh_omp >= 0 ? std::to_string(accuracy_bvh_omp) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bvh_omp_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bvh_omp >= 0 ? std::to_string(accuracy_bvh_omp) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_bvh_omp, n, log_output);
            print_validation_forces(forces_bvh_omp, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // BVH Parlay
        log_output << "BVH ParlayLib parallel approach:" << std::endl;
        log_output << "Using " << parlay::num_workers() << " workers..." << std::endl;
        std::cout << "BVH ParlayLib parallel approach:" << std::endl;
        std::cout << "Using " << parlay::num_workers() << " workers..." << std::endl;
        
        parlay::sequence<Vector<D>> forces_bvh_parlay;
        auto time_bvh_parlay = safely_execute(log_output, "BVH_Parlay", [&]() {
            forces_bvh_parlay = bvh_parlay_n_body<D>(parlay_bodies);
            return forces_bvh_parlay;
        });
        
        double accuracy_bvh_parlay = -1.0;
        if (calculate_accuracy) {
            accuracy_bvh_parlay = compute_accuracy(std::vector<Vector<D>>(forces_bvh_parlay.begin(), forces_bvh_parlay.end()), reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_bvh_parlay_seconds = time_bvh_parlay / 1e6;
        
        if (time_bvh_parlay >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "BVH_Parlay," << n << "," << D;
            if (time_bvh_parlay_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_bvh_parlay_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_bvh_parlay_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_bvh_parlay;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_bvh_parlay_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_bvh_parlay >= 0 ? std::to_string(accuracy_bvh_parlay) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_bvh_parlay_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_bvh_parlay >= 0 ? std::to_string(accuracy_bvh_parlay) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(std::vector<Vector<D>>(forces_bvh_parlay.begin(), forces_bvh_parlay.end()), n, log_output);
            print_validation_forces(std::vector<Vector<D>>(forces_bvh_parlay.begin(), forces_bvh_parlay.end()), n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
    }
    
    // FMM methods - only run if specified
    if (run_fmm) {
        // FMM sequential
        log_output << "FMM sequential approach:" << std::endl;
        std::cout << "FMM sequential approach:" << std::endl;
        
        std::vector<Vector<D>> forces_fmm_seq;
        auto time_fmm_seq = safely_execute(log_output, "FMM_Sequential", [&]() {
            forces_fmm_seq = fmm_seq_n_body<D>(bodies);
            return forces_fmm_seq;
        });
        
        double accuracy_fmm_seq = -1.0;
        if (calculate_accuracy) {
            accuracy_fmm_seq = compute_accuracy(forces_fmm_seq, reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_fmm_seq_seconds = time_fmm_seq / 1e6;
        
        if (time_fmm_seq >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "FMM_Sequential," << n << "," << D;
            if (time_fmm_seq_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_fmm_seq_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_fmm_seq_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_fmm_seq;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_fmm_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_fmm_seq >= 0 ? std::to_string(accuracy_fmm_seq) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_fmm_seq_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_fmm_seq >= 0 ? std::to_string(accuracy_fmm_seq) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_fmm_seq, n, log_output);
            print_validation_forces(forces_fmm_seq, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // FMM OpenMP
        log_output << "FMM OpenMP parallel approach:" << std::endl;
        log_output << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        std::cout << "FMM OpenMP parallel approach:" << std::endl;
        std::cout << "Using " << omp_get_max_threads() << " threads..." << std::endl;
        
        std::vector<Vector<D>> forces_fmm_omp;
        auto time_fmm_omp = safely_execute(log_output, "FMM_OpenMP", [&]() {
            forces_fmm_omp = fmm_omp_n_body<D>(bodies);
            return forces_fmm_omp;
        });
        
        double accuracy_fmm_omp = -1.0;
        if (calculate_accuracy) {
            accuracy_fmm_omp = compute_accuracy(forces_fmm_omp, reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_fmm_omp_seconds = time_fmm_omp / 1e6;
        
        if (time_fmm_omp >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "FMM_OpenMP," << n << "," << D;
            if (time_fmm_omp_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_fmm_omp_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_fmm_omp_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_fmm_omp;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_fmm_omp_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_fmm_omp >= 0 ? std::to_string(accuracy_fmm_omp) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_fmm_omp_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_fmm_omp >= 0 ? std::to_string(accuracy_fmm_omp) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(forces_fmm_omp, n, log_output);
            print_validation_forces(forces_fmm_omp, n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
        
        // FMM Parlay
        log_output << "FMM ParlayLib parallel approach:" << std::endl;
        log_output << "Using " << parlay::num_workers() << " workers..." << std::endl;
        std::cout << "FMM ParlayLib parallel approach:" << std::endl;
        std::cout << "Using " << parlay::num_workers() << " workers..." << std::endl;
        
        parlay::sequence<Vector<D>> forces_fmm_parlay;
        auto time_fmm_parlay = safely_execute(log_output, "FMM_Parlay", [&]() {
            forces_fmm_parlay = fmm_parlay_n_body<D>(parlay_bodies);
            return forces_fmm_parlay;
        });
        
        double accuracy_fmm_parlay = -1.0;
        if (calculate_accuracy) {
            accuracy_fmm_parlay = compute_accuracy(std::vector<Vector<D>>(forces_fmm_parlay.begin(), forces_fmm_parlay.end()), reference_forces);
        }
        
        // Convert time from microseconds to seconds
        double time_fmm_parlay_seconds = time_fmm_parlay / 1e6;
        
        if (time_fmm_parlay >= 0) {
            // Write to CSV (formatting time in seconds with 6 decimal places or scientific notation)
            csv_output << "FMM_Parlay," << n << "," << D;
            if (time_fmm_parlay_seconds < 1e-6) {
                // Use scientific notation for very small times
                csv_output << "," << std::scientific << std::setprecision(6) << time_fmm_parlay_seconds;
            } else {
                // Use fixed notation with 6 decimal places
                csv_output << "," << std::fixed << std::setprecision(6) << time_fmm_parlay_seconds;
            }
            if (calculate_accuracy) {
                csv_output << "," << std::fixed << std::setprecision(2) << accuracy_fmm_parlay;
            }
            csv_output << std::endl;
            
            // Write to log
            log_output << "Time taken: " << time_fmm_parlay_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                log_output << "Accuracy: " << (accuracy_fmm_parlay >= 0 ? std::to_string(accuracy_fmm_parlay) + "%" : "Not calculated") << std::endl;
            }
            std::cout << "Time taken: " << time_fmm_parlay_seconds << " s" << std::endl;
            if (calculate_accuracy) {
                std::cout << "Accuracy: " << (accuracy_fmm_parlay >= 0 ? std::to_string(accuracy_fmm_parlay) + "%" : "Not calculated") << std::endl;
            }
            print_validation_forces(std::vector<Vector<D>>(forces_fmm_parlay.begin(), forces_fmm_parlay.end()), n, log_output);
            print_validation_forces(std::vector<Vector<D>>(forces_fmm_parlay.begin(), forces_fmm_parlay.end()), n, std::cout);
        }
        log_output << std::endl;
        std::cout << std::endl;
    }
    
    csv_output.close();
    log_output.close();
    
    std::cout << "Benchmark complete. Results written to:" << std::endl;
    std::cout << " - " << csv_file << std::endl;
    std::cout << " - " << out_file << std::endl;
}

int main(int argc, char* argv[]) {
    int dimension = 3;  // Default dimension (3D)
    int num_bodies = 1000;  // Default number of bodies
    bool calculate_accuracy = false;  // Default: accuracy calculation is OFF
    std::string methods = "";  // Default: run all methods based on body count
    bool override_bf_limit = false;  // Whether to override the brute force limit
    
    // Parse command line arguments with new format
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--dim") && i + 1 < argc) {
            dimension = std::stoi(argv[++i]);
            if (dimension != 2 && dimension != 3) {
                std::cerr << "Error: Dimension must be either 2 or 3" << std::endl;
                return 1;
            }
        } else if ((arg == "-N" || arg == "--bodies") && i + 1 < argc) {
            num_bodies = std::stoi(argv[++i]);
            if (num_bodies <= 0) {
                std::cerr << "Error: Number of bodies must be positive" << std::endl;
                return 1;
            }
        } else if ((arg == "-a" || arg == "--accuracy") && i + 1 < argc) {
            int acc_flag = std::stoi(argv[++i]);
            calculate_accuracy = (acc_flag == 1);
        } else if ((arg == "-m" || arg == "--methods") && i + 1 < argc) {
            methods = argv[++i];
            // If only brute force is specified, allow overriding the size limit
            if (methods == "a") {
                override_bf_limit = true;
            }
            // Validate method characters
            for (char c : methods) {
                if (c != 'a' && c != 'b' && c != 'h' && c != 'f') {
                    std::cerr << "Error: Invalid method '" << c << "'" << std::endl;
                    std::cerr << "Valid methods: a=bruteforce, b=barnes-hut, h=bvh, f=fmm" << std::endl;
                    return 1;
                }
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -d, --dim <2|3>     Set simulation dimension (default: 3)" << std::endl;
            std::cout << "  -N, --bodies <num>  Set number of bodies (default: 1000)" << std::endl;
            std::cout << "  -a, --accuracy <0|1> Enable accuracy calculation (default: 0 - OFF)" << std::endl;
            std::cout << "  -m, --methods <str> Specify which methods to run (default: all)" << std::endl;
            std::cout << "                      a=bruteforce, b=barnes-hut, h=hilbert bvh, f=fmm" << std::endl;
            std::cout << "                      Example: -m bf runs Barnes-Hut and FMM only" << std::endl;
            std::cout << "  -h, --help          Display this help message" << std::endl;
            return 0;
        }
    }

    // Get run ID from current date and time
    std::string run_id = get_run_id();
    
    // Generate bodies and run benchmarks based on dimension
    try {
        if (dimension == 2) {
            auto bodies = generate_random_bodies<2>(num_bodies);
            run_benchmark<2>(bodies, run_id, calculate_accuracy, methods, override_bf_limit);
        } else {
            auto bodies = generate_random_bodies<3>(num_bodies);
            run_benchmark<3>(bodies, run_id, calculate_accuracy, methods, override_bf_limit);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
    
    return 0;
}
