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
#include "utils.h"  // Include utils.h before methods.h
#include "methods.h"

// Helper for running and timing different n-body methods
template <int D>
void run_benchmark(const std::vector<Body<D>>& bodies, const std::string& run_id) {
    int n = bodies.size();
    
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
    
    // Write headers to CSV
    csv_output << "Method,Bodies,Time(us)" << std::endl;
    
    // Write benchmark information to log
    log_output << "Running N-body simulation benchmark with:" << std::endl;
    log_output << "  Dimension: " << D << "D" << std::endl;
    log_output << "  Bodies: " << n << std::endl;
    log_output << "  Run ID: " << run_id << std::endl;
    log_output << std::endl;
    
    std::cout << "Running N-body simulation benchmark with:" << std::endl;
    std::cout << "  Dimension: " << D << "D" << std::endl;
    std::cout << "  Bodies: " << n << std::endl;
    std::cout << "  Run ID: " << run_id << std::endl;
    std::cout << std::endl;

    // Brute force sequential
    log_output << "Brute force O(n²) sequential approach:" << std::endl;
    std::cout << "Brute force O(n²) sequential approach:" << std::endl;
    
    std::vector<Vector<D>> forces_bf_seq;
    auto time_bf_seq = safely_execute(log_output, "BruteForce_Sequential", [&]() {
        forces_bf_seq = brute_force_seq_n_body<D>(bodies);
        return forces_bf_seq;
    });
    
    if (time_bf_seq >= 0) {
        csv_output << "BruteForce_Sequential," << n << "," << time_bf_seq << std::endl;
        log_output << "Time taken: " << time_bf_seq / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bf_seq / 1e6 << " s" << std::endl;
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
    
    if (time_bf_omp1 >= 0) {
        csv_output << "BruteForce_OpenMP1," << n << "," << time_bf_omp1 << std::endl;
        log_output << "Time taken: " << time_bf_omp1 / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bf_omp1 / 1e6 << " s" << std::endl;
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
    
    if (time_bf_omp2 >= 0) {
        csv_output << "BruteForce_OpenMP2," << n << "," << time_bf_omp2 << std::endl;
        log_output << "Time taken: " << time_bf_omp2 / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bf_omp2 / 1e6 << " s" << std::endl;
        print_validation_forces(forces_bf_omp2, n, log_output);
        print_validation_forces(forces_bf_omp2, n, std::cout);
    }
    log_output << std::endl;
    std::cout << std::endl;
    
    // Convert to parlay sequence for parlay methods
    parlay::sequence<Body<D>> parlay_bodies(bodies.begin(), bodies.end());
    
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
    
    if (time_bf_parlay1 >= 0) {
        csv_output << "BruteForce_Parlay1," << n << "," << time_bf_parlay1 << std::endl;
        log_output << "Time taken: " << time_bf_parlay1 / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bf_parlay1 / 1e6 << " s" << std::endl;
        print_validation_forces(forces_bf_parlay1, n, log_output);
        print_validation_forces(forces_bf_parlay1, n, std::cout);
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
    
    if (time_bf_parlay2 >= 0) {
        csv_output << "BruteForce_Parlay2," << n << "," << time_bf_parlay2 << std::endl;
        log_output << "Time taken: " << time_bf_parlay2 / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bf_parlay2 / 1e6 << " s" << std::endl;
        print_validation_forces(forces_bf_parlay2, n, log_output);
        print_validation_forces(forces_bf_parlay2, n, std::cout);
    }
    log_output << std::endl;
    std::cout << std::endl;
    
    // Barnes-Hut sequential
    log_output << "Barnes-Hut sequential approach:" << std::endl;
    std::cout << "Barnes-Hut sequential approach:" << std::endl;
    
    std::vector<Vector<D>> forces_bh_seq;
    auto time_bh_seq = safely_execute(log_output, "BarnesHut_Sequential", [&]() {
        forces_bh_seq = barnes_hut_seq_n_body<D>(bodies);
        return forces_bh_seq;
    });
    
    if (time_bh_seq >= 0) {
        csv_output << "BarnesHut_Sequential," << n << "," << time_bh_seq << std::endl;
        log_output << "Time taken: " << time_bh_seq / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bh_seq / 1e6 << " s" << std::endl;
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
    
    if (time_bh_omp >= 0) {
        csv_output << "BarnesHut_OpenMP," << n << "," << time_bh_omp << std::endl;
        log_output << "Time taken: " << time_bh_omp / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bh_omp / 1e6 << " s" << std::endl;
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
    
    if (time_bh_parlay >= 0) {
        csv_output << "BarnesHut_Parlay," << n << "," << time_bh_parlay << std::endl;
        log_output << "Time taken: " << time_bh_parlay / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bh_parlay / 1e6 << " s" << std::endl;
        print_validation_forces(forces_bh_parlay, n, log_output);
        print_validation_forces(forces_bh_parlay, n, std::cout);
    }
    log_output << std::endl;
    std::cout << std::endl;

    // BVH sequential
    log_output << "BVH sequential approach:" << std::endl;
    std::cout << "BVH sequential approach:" << std::endl;
    
    std::vector<Vector<D>> forces_bvh_seq;
    auto time_bvh_seq = safely_execute(log_output, "BVH_Sequential", [&]() {
        forces_bvh_seq = bvh_seq_n_body<D>(bodies);
        return forces_bvh_seq;
    });
    
    if (time_bvh_seq >= 0) {
        csv_output << "BVH_Sequential," << n << "," << time_bvh_seq << std::endl;
        log_output << "Time taken: " << time_bvh_seq / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bvh_seq / 1e6 << " s" << std::endl;
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
    
    if (time_bvh_omp >= 0) {
        csv_output << "BVH_OpenMP," << n << "," << time_bvh_omp << std::endl;
        log_output << "Time taken: " << time_bvh_omp / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bvh_omp / 1e6 << " s" << std::endl;
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
    
    if (time_bvh_parlay >= 0) {
        csv_output << "BVH_Parlay," << n << "," << time_bvh_parlay << std::endl;
        log_output << "Time taken: " << time_bvh_parlay / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_bvh_parlay / 1e6 << " s" << std::endl;
        print_validation_forces(forces_bvh_parlay, n, log_output);
        print_validation_forces(forces_bvh_parlay, n, std::cout);
    }
    log_output << std::endl;
    std::cout << std::endl;
    
    // FMM sequential
    log_output << "FMM sequential approach:" << std::endl;
    std::cout << "FMM sequential approach:" << std::endl;
    
    std::vector<Vector<D>> forces_fmm_seq;
    auto time_fmm_seq = safely_execute(log_output, "FMM_Sequential", [&]() {
        forces_fmm_seq = fmm_seq_n_body<D>(bodies);
        return forces_fmm_seq;
    });
    
    if (time_fmm_seq >= 0) {
        csv_output << "FMM_Sequential," << n << "," << time_fmm_seq << std::endl;
        log_output << "Time taken: " << time_fmm_seq / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_fmm_seq / 1e6 << " s" << std::endl;
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
    
    if (time_fmm_omp >= 0) {
        csv_output << "FMM_OpenMP," << n << "," << time_fmm_omp << std::endl;
        log_output << "Time taken: " << time_fmm_omp / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_fmm_omp / 1e6 << " s" << std::endl;
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
    
    if (time_fmm_parlay >= 0) {
        csv_output << "FMM_Parlay," << n << "," << time_fmm_parlay << std::endl;
        log_output << "Time taken: " << time_fmm_parlay / 1e6 << " s" << std::endl;
        std::cout << "Time taken: " << time_fmm_parlay / 1e6 << " s" << std::endl;
        print_validation_forces(forces_fmm_parlay, n, log_output);
        print_validation_forces(forces_fmm_parlay, n, std::cout);
    }
    log_output << std::endl;
    std::cout << std::endl;
    
    csv_output.close();
    log_output.close();
    
    std::cout << "Benchmark complete. Results written to:" << std::endl;
    std::cout << " - " << csv_file << std::endl;
    std::cout << " - " << out_file << std::endl;
}

int main(int argc, char* argv[]) {
    int dimension = 3;  // Default dimension (3D)
    int num_bodies = 1000;  // Default number of bodies
    
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
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -d, --dim <2|3>     Set simulation dimension (default: 3)" << std::endl;
            std::cout << "  -N, --bodies <num>  Set number of bodies (default: 1000)" << std::endl;
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
            run_benchmark<2>(bodies, run_id);
        } else {
            auto bodies = generate_random_bodies<3>(num_bodies);
            run_benchmark<3>(bodies, run_id);
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
