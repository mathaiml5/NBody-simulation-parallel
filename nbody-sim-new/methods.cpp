#include "methods.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

// Brute force sequential implementation
template <int D>
std::vector<Vector<D>> brute_force_seq_n_body(const std::vector<Body<D>>& bodies) {
    const size_t n = bodies.size();
    std::vector<Vector<D>> forces(n);
    
    // Initialize forces to zero
    for (size_t i = 0; i < n; ++i) {
        forces[i] = Vector<D>();
    }
    
    // Calculate forces with symmetry
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            // Compute distance vector
            Vector<D> diff = bodies[j].position - bodies[i].position;
            double dist_sq = diff.magnitude_squared();
            
            if (dist_sq < 1e-10) continue; // Avoid division by zero
            
            double dist = std::sqrt(dist_sq);
            double dist_cb = dist_sq * dist;
            
            // Calculate force magnitude
            double force_mag = G * bodies[i].mass * bodies[j].mass / dist_cb;
            
            // Force direction is along the direction vector
            Vector<D> force = diff.normalized() * force_mag;
            
            // Apply forces with opposite directions (Newton's third law)
            forces[j] += force;
            forces[i] -= force;
        }
    }
    
    return forces;
}

// OpenMP brute force implementation (parallelize outer loop with local storage)
template <int D>
std::vector<Vector<D>> brute_force_omp_n_body_1(const std::vector<Body<D>>& bodies) {
    const size_t n = bodies.size();
    std::vector<Vector<D>> forces(n);
    
    // Get the number of threads
    int num_threads = omp_get_max_threads();
    
    // Create thread-local storage
    std::vector<std::vector<Vector<D>>> local(num_threads, std::vector<Vector<D>>(n));
    
    // Compute iterations of outer loop in parallel
    // Threads work on local copies
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        
        #pragma omp for
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                // Compute distance vector
                Vector<D> diff = bodies[j].position - bodies[i].position;
                double dist_sq = diff.magnitude_squared();
                
                if (dist_sq < 1e-10) continue; // Avoid division by zero
                
                double dist = std::sqrt(dist_sq);
                double dist_cb = dist_sq * dist;
                
                // Calculate force magnitude
                double force_mag = G * bodies[i].mass * bodies[j].mass / dist_cb;
                
                // Force direction is along the direction vector
                Vector<D> force = diff.normalized() * force_mag;
                
                // Apply forces to local storage
                local[t][j] += force;
                local[t][i] -= force;
            }
        }
    }
    
    // Combine forces from all threads
    for (int t = 0; t < num_threads; t++) {
        for (size_t i = 0; i < n; i++) {
            forces[i] += local[t][i];
        }
    }
    
    return forces;
}

// OpenMP brute force implementation (memory-efficient)
template <int D>
std::vector<Vector<D>> brute_force_omp_n_body_2(const std::vector<Body<D>>& bodies) {
    const size_t n = bodies.size();
    std::vector<Vector<D>> forces(n);
    
    // Initialize forces to zero
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        forces[i] = Vector<D>();
    }
    
    // Compute forces with all to all approach
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            
            // Compute distance vector
            Vector<D> diff = bodies[j].position - bodies[i].position;
            double dist_sq = diff.magnitude_squared();
            
            if (dist_sq < 1e-10) continue; // Avoid division by zero
            
            double dist = std::sqrt(dist_sq);
            double dist_cb = dist_sq * dist;
            
            // Calculate force magnitude
            double force_mag = G * bodies[i].mass * bodies[j].mass / dist_cb;
            
            // Force direction is along the direction vector but opposite to i
            Vector<D> force = diff.normalized() * force_mag;
            
            // Apply force to body i (opposite direction)
            forces[i] -= force;
        }
    }
    
    return forces;
}

// ParlayLib brute force implementation (memory-intensive with thread-local storage)
template <int D>
parlay::sequence<Vector<D>> brute_force_parlay_n_body_1(const parlay::sequence<Body<D>>& bodies) {
    const size_t n = bodies.size();
    int num_workers = parlay::num_workers();
    
    // Thread-local storage
    auto local = parlay::tabulate(num_workers, [n](size_t) {
        return parlay::sequence<Vector<D>>(n);
    });
    
    // Compute grain size based on problem size and number of workers
    size_t grain_size = std::max<size_t>(1, n / (4 * num_workers));
    
    // Compute forces in parallel
    parlay::parallel_for(0, n, [&](size_t i) {
        size_t worker_id = parlay::worker_id();
        for (size_t j = i + 1; j < n; j++) {
            // Compute distance vector
            Vector<D> diff = bodies[j].position - bodies[i].position;
            double dist_sq = diff.magnitude_squared();
            
            if (dist_sq < 1e-10) continue; // Avoid division by zero
            
            double dist = std::sqrt(dist_sq);
            double dist_cb = dist_sq * dist;
            
            // Calculate force magnitude
            double force_mag = G * bodies[i].mass * bodies[j].mass / dist_cb;
            
            // Force direction is along the direction vector
            Vector<D> force = diff.normalized() * force_mag;
            
            // Apply forces to local storage
            local[worker_id][j] += force;
            local[worker_id][i] -= force;
        }
    }, grain_size);
    
    // Combine forces from all workers
    auto forces = parlay::sequence<Vector<D>>(n);
    for (int t = 0; t < num_workers; t++) {
        for (size_t i = 0; i < n; i++) {
            forces[i] += local[t][i];
        }
    }
    
    return forces;
}

// ParlayLib brute force implementation (memory-efficient)
template <int D>
parlay::sequence<Vector<D>> brute_force_parlay_n_body_2(const parlay::sequence<Body<D>>& bodies) {
    const size_t n = bodies.size();
    auto forces = parlay::sequence<Vector<D>>(n);
    int num_workers = parlay::num_workers();
    
    // Compute grain size based on problem size and number of workers
    size_t grain_size = std::max<size_t>(1, n / (4 * num_workers));
    
    // Compute forces with all to all approach
    parlay::parallel_for(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            
            // Compute distance vector
            Vector<D> diff = bodies[j].position - bodies[i].position;
            double dist_sq = diff.magnitude_squared();
            
            if (dist_sq < 1e-10) continue; // Avoid division by zero
            
            double dist = std::sqrt(dist_sq);
            double dist_cb = dist_sq * dist;
            
            // Calculate force magnitude
            double force_mag = G * bodies[i].mass * bodies[j].mass / dist_cb;
            
            // Force direction is along the direction vector but opposite to i
            Vector<D> force = diff.normalized() * force_mag;
            
            // Apply force to body i (opposite direction)
            forces[i] -= force;
        }
    }, grain_size);
    
    return forces;
}

// Barnes-Hut sequential implementation
template <int D>
std::vector<Vector<D>> barnes_hut_seq_n_body(const std::vector<Body<D>>& bodies, double theta) {
    // Build the octree - ignore theta parameter, use global constant
    Octree<D> octree(bodies);
    
    // Calculate forces using Barnes-Hut algorithm
    return octree.calculate_forces(bodies);
}

// Barnes-Hut OpenMP implementation
template <int D>
std::vector<Vector<D>> barnes_hut_omp_n_body(const std::vector<Body<D>>& bodies, double theta) {
    // Build the octree (sequential part) - ignore theta parameter, use global constant
    Octree<D> octree(bodies);
    
    // Parallel force calculation
    const size_t n = bodies.size();
    std::vector<Vector<D>> forces(n);
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        forces[i] = octree.root->calculate_force(bodies[i], BARNES_HUT_THETA);
    }
    
    return forces;
}

// Barnes-Hut ParlayLib implementation
template <int D>
parlay::sequence<Vector<D>> barnes_hut_parlay_n_body(const parlay::sequence<Body<D>>& bodies, double theta) {
    // Convert parlay sequence to std::vector for octree building
    std::vector<Body<D>> std_bodies(bodies.begin(), bodies.end());
    
    // Build the octree (sequential part) - ignore theta parameter, use global constant
    Octree<D> octree(std_bodies);
    
    // Parallel force calculation
    const size_t n = bodies.size();
    parlay::sequence<Vector<D>> forces(n);
    
    parlay::parallel_for(0, n, [&](size_t i) {
        forces[i] = octree.root->calculate_force(std_bodies[i], BARNES_HUT_THETA);
    });
    
    return forces;
}

// FMM sequential implementation
template <int D>
std::vector<Vector<D>> fmm_seq_n_body(const std::vector<Body<D>>& bodies, 
                                     int max_bodies_per_leaf, 
                                     int max_level, 
                                     int order) {
    // Use higher order for better accuracy (minimum 10 for good results)
    order = std::max(order, 10);
    
    // Create FMM instance with proper parameters
    FMM<D> fmm(bodies, max_bodies_per_leaf, max_level, order);
    
    // Calculate forces by calling calculate_accurate_force for each body
    const size_t n = bodies.size();
    std::vector<Vector<D>> forces(n, Vector<D>());
    
    for (size_t i = 0; i < n; ++i) {
        forces[i] = fmm.calculate_accurate_force(bodies[i]);
    }
    
    return forces;
}

// Helper function to find the leaf node containing a specific body
template <int D>
FMMNode<D>* find_leaf_containing_body(FMMNode<D>* node, const Body<D>& body) {
    if (!node) return nullptr;
    
    // Check if this is a leaf node
    if (node->is_leaf()) {
        // Check if the body is in this leaf
        for (const Body<D>* node_body : node->bodies) {
            // Compare positions to identify the body
            bool same_body = true;
            for (int d = 0; d < D; ++d) {
                if (std::abs(node_body->position[d] - body.position[d]) > 1e-9) {
                    same_body = false;
                    break;
                }
            }
            if (same_body) return node;
        }
        return nullptr; // Body not found in this leaf
    }
    
    // For internal nodes, determine which child might contain the body
    // Based on the body's position relative to the node's center
    int child_idx = 0;
    for (int d = 0; d < D; ++d) {
        if (body.position[d] > node->center[d]) {
            child_idx |= (1 << d);
        }
    }
    
    // Check if that child exists and search in it
    if (node->children[child_idx]) {
        return find_leaf_containing_body(node->children[child_idx].get(), body);
    }
    
    // If we get here, the body is not in the tree
    return nullptr;
}

// FMM OpenMP implementation
template <int D>
std::vector<Vector<D>> fmm_omp_n_body(const std::vector<Body<D>>& bodies, 
                                     int max_bodies_per_leaf, 
                                     int max_level, 
                                     int order) {
    // Use higher order for better accuracy
    order = std::max(order, 8);

    // Use optimized OpenMP-specific FMM implementation
    FMM_OMP<D> fmm_omp(bodies, max_bodies_per_leaf, max_level, order);

    // Use the optimized implementation to calculate forces
    return fmm_omp.calculate_forces(bodies);
}

// FMM ParlayLib implementation - optimized version
template <int D>
parlay::sequence<Vector<D>> fmm_parlay_n_body(const parlay::sequence<Body<D>>& bodies, 
                                             int max_bodies_per_leaf, 
                                             int max_level, 
                                             int order) {
    // Use higher order for better accuracy
    order = std::max(order, 10);
    
    // Tune parameters for better balance between speed and accuracy
    max_bodies_per_leaf = std::min(max_bodies_per_leaf, 32); // Smaller leaf nodes
    max_level = std::max(max_level, 8); // Ensure enough depth
    
    // Create optimized Parlay-specific FMM implementation
    FMM_Parlay<D> fmm_parlay(bodies, max_bodies_per_leaf, max_level, order);
    
    // Use the optimized implementation to calculate forces
    auto forces = fmm_parlay.calculate_forces(bodies);
    
    return forces;
}

// BVH sequential implementation
template <int D>
std::vector<Vector<D>> bvh_seq_n_body(const std::vector<Body<D>>& bodies, 
                                     int max_bodies_per_leaf) {
    // Create BVH instance and calculate forces
    // Use default theta value of 0.5
    BVH<D> bvh(bodies, max_bodies_per_leaf);
    return bvh.calculate_forces(bodies);
}

// BVH OpenMP implementation
template <int D>
std::vector<Vector<D>> bvh_omp_n_body(const std::vector<Body<D>>& bodies, int max_bodies_per_leaf) {
    // Create BVH instance
    // Use default theta value of 0.5
    BVH<D> bvh(bodies, max_bodies_per_leaf);
    
    // Parallel force calculation
    const size_t n = bodies.size();
    std::vector<Vector<D>> forces(n);
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        forces[i] = bvh.calculate_force(bodies[i], bvh.root.get());
    }
    
    return forces;
}

// BVH ParlayLib implementation
template <int D>
parlay::sequence<Vector<D>> bvh_parlay_n_body(const parlay::sequence<Body<D>>& bodies, int max_bodies_per_leaf) {
    // Convert parlay sequence to std::vector
    std::vector<Body<D>> std_bodies(bodies.begin(), bodies.end());
    
    // Create BVH instance
    // Use default theta value of 0.5
    BVH<D> bvh(std_bodies, max_bodies_per_leaf);
    
    // Parallel force calculation
    const size_t n = bodies.size();
    parlay::sequence<Vector<D>> forces(n);
    
    parlay::parallel_for(0, n, [&](size_t i) {
        forces[i] = bvh.calculate_force(std_bodies[i], bvh.root.get());
    });
    
    return forces;
}

// Helper to update body velocities
template <int D>
void update_body_velocities(std::vector<Body<D>>& bodies, 
                           const std::vector<Vector<D>>& forces,
                           double dt) {
    const size_t n = bodies.size();
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        // F = ma -> a = F/m
        // v = v0 + a*dt
        bodies[i].velocity += forces[i] / bodies[i].mass * dt;
    }
}

// Helper to update body positions
template <int D>
void update_body_positions(std::vector<Body<D>>& bodies, double dt) {
    const size_t n = bodies.size();
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        // x = x0 + v*dt
        bodies[i].position += bodies[i].velocity * dt;
    }
}

// Explicit template instantiations for common dimensions
template std::vector<Vector<2>> brute_force_seq_n_body<2>(const std::vector<Body<2>>& bodies);
template std::vector<Vector<3>> brute_force_seq_n_body<3>(const std::vector<Body<3>>& bodies);

template std::vector<Vector<2>> brute_force_omp_n_body_1<2>(const std::vector<Body<2>>& bodies);
template std::vector<Vector<3>> brute_force_omp_n_body_1<3>(const std::vector<Body<3>>& bodies);

template std::vector<Vector<2>> brute_force_omp_n_body_2<2>(const std::vector<Body<2>>& bodies);
template std::vector<Vector<3>> brute_force_omp_n_body_2<3>(const std::vector<Body<3>>& bodies);

template parlay::sequence<Vector<2>> brute_force_parlay_n_body_1<2>(const parlay::sequence<Body<2>>& bodies);
template parlay::sequence<Vector<3>> brute_force_parlay_n_body_1<3>(const parlay::sequence<Body<3>>& bodies);

template parlay::sequence<Vector<2>> brute_force_parlay_n_body_2<2>(const parlay::sequence<Body<2>>& bodies);
template parlay::sequence<Vector<3>> brute_force_parlay_n_body_2<3>(const parlay::sequence<Body<3>>& bodies);

template std::vector<Vector<2>> barnes_hut_seq_n_body<2>(const std::vector<Body<2>>& bodies, double theta);
template std::vector<Vector<3>> barnes_hut_seq_n_body<3>(const std::vector<Body<3>>& bodies, double theta);

template std::vector<Vector<2>> barnes_hut_omp_n_body<2>(const std::vector<Body<2>>& bodies, double theta);
template std::vector<Vector<3>> barnes_hut_omp_n_body<3>(const std::vector<Body<3>>& bodies, double theta);

template parlay::sequence<Vector<2>> barnes_hut_parlay_n_body<2>(const parlay::sequence<Body<2>>& bodies, double theta);
template parlay::sequence<Vector<3>> barnes_hut_parlay_n_body<3>(const parlay::sequence<Body<3>>& bodies, double theta);

template std::vector<Vector<2>> bvh_seq_n_body<2>(const std::vector<Body<2>>& bodies, int max_bodies_per_leaf);
template std::vector<Vector<3>> bvh_seq_n_body<3>(const std::vector<Body<3>>& bodies, int max_bodies_per_leaf);

template std::vector<Vector<2>> bvh_omp_n_body<2>(const std::vector<Body<2>>& bodies, int max_bodies_per_leaf);
template std::vector<Vector<3>> bvh_omp_n_body<3>(const std::vector<Body<3>>& bodies, int max_bodies_per_leaf);

template parlay::sequence<Vector<2>> bvh_parlay_n_body<2>(const parlay::sequence<Body<2>>& bodies, int max_bodies_per_leaf);
template parlay::sequence<Vector<3>> bvh_parlay_n_body<3>(const parlay::sequence<Body<3>>& bodies, int max_bodies_per_leaf);

template std::vector<Vector<2>> fmm_seq_n_body<2>(const std::vector<Body<2>>& bodies, int max_bodies_per_leaf, int max_level, int order);
template std::vector<Vector<3>> fmm_seq_n_body<3>(const std::vector<Body<3>>& bodies, int max_bodies_per_leaf, int max_level, int order);

template std::vector<Vector<2>> fmm_omp_n_body<2>(const std::vector<Body<2>>& bodies, int max_bodies_per_leaf, int max_level, int order);
template std::vector<Vector<3>> fmm_omp_n_body<3>(const std::vector<Body<3>>& bodies, int max_bodies_per_leaf, int max_level, int order);

template parlay::sequence<Vector<2>> fmm_parlay_n_body<2>(const parlay::sequence<Body<2>>& bodies, int max_bodies_per_leaf, int max_level, int order);
template parlay::sequence<Vector<3>> fmm_parlay_n_body<3>(const parlay::sequence<Body<3>>& bodies, int max_bodies_per_leaf, int max_level, int order);

template void update_body_velocities<2>(std::vector<Body<2>>& bodies, const std::vector<Vector<2>>& forces, double dt);
template void update_body_velocities<3>(std::vector<Body<3>>& bodies, const std::vector<Vector<3>>& forces, double dt);

template void update_body_positions<2>(std::vector<Body<2>>& bodies, double dt);
template void update_body_positions<3>(std::vector<Body<3>>& bodies, double dt);
