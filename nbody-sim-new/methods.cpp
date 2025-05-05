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
    // Use higher order for better accuracy (minimum 8 for good results)
    order = std::max(order, 8);
    
    // Create FMM instance with proper parameters
    FMM<D> fmm(bodies, max_bodies_per_leaf, max_level, order);
    
    // Calculate forces
    return fmm.calculate_forces(bodies);
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
    // Use higher order for better accuracy (minimum 8 for good results)
    order = std::max(order, 8);
    
    // Create FMM instance and build tree sequentially
    FMM<D> fmm(bodies, max_bodies_per_leaf, max_level, order);
    
    // Get number of bodies
    const size_t n = bodies.size();
    std::vector<Vector<D>> forces(n, Vector<D>());
    
    // Upward pass (sequential due to dependencies)
    fmm.upward_pass();
    
    // Collect all nodes for parallel processing in the interaction pass
    std::vector<FMMNode<D>*> all_nodes;
    
    // Recursive function to collect all nodes
    std::function<void(FMMNode<D>*)> collect_nodes;
    collect_nodes = [&all_nodes, &collect_nodes](FMMNode<D>* node) {
        if (!node) return;
        all_nodes.push_back(node);
        for (const auto& child : node->children) {
            if (child) collect_nodes(child.get());
        }
    };
    
    // Fill all_nodes
    collect_nodes(fmm.root.get());
    
    // Parallel M2L operations
    #pragma omp parallel for
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        FMMNode<D>* node = all_nodes[i];
        for (FMMNode<D>* interaction : node->interaction_list) {
            if (interaction) node->translate_multipole_to_local(interaction, order);
        }
    }
    
    // Sequential downward pass for L2L (due to dependencies)
    // Fix: Properly capture the recursive lambda
    std::function<void(FMMNode<D>*)> translate_local;
    translate_local = [&order, &translate_local](FMMNode<D>* node) {
        if (!node) return;
        
        // Translate to children
        if (!node->is_leaf()) {
            node->translate_local_to_children(order);
            
            // Continue down the tree
            for (auto& child : node->children) {
                if (child) translate_local(child.get());
            }
        }
    };
    
    translate_local(fmm.root.get());
    
    // Collect leaf nodes for parallel processing
    std::vector<FMMNode<D>*> leaf_nodes;
    std::function<void(FMMNode<D>*)> collect_leaves;
    collect_leaves = [&leaf_nodes, &collect_leaves](FMMNode<D>* node) {
        if (!node) return;
        if (node->is_leaf()) {
            leaf_nodes.push_back(node);
        } else {
            for (const auto& child : node->children) {
                if (child) collect_leaves(child.get());
            }
        }
    };
    
    collect_leaves(fmm.root.get());
    
    // Map bodies to their leaf nodes for direct lookup
    std::vector<FMMNode<D>*> body_to_leaf(n, nullptr);
    for (FMMNode<D>* leaf : leaf_nodes) {
        for (Body<D>* body_ptr : leaf->bodies) {
            // Find the index of this body in the original array
            for (size_t i = 0; i < n; ++i) {
                if (&bodies[i] == body_ptr) {
                    body_to_leaf[i] = leaf;
                    break;
                }
            }
        }
    }
    
    // Parallel L2P and direct calculations
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        const Body<D>& body = bodies[i];
        FMMNode<D>* leaf = body_to_leaf[i];
        
        if (leaf) {
            // Direct calculations with neighbors
            for (FMMNode<D>* neighbor : leaf->neighbor_list) {
                if (!neighbor) continue;
                for (const Body<D>* other : neighbor->bodies) {
                    if (!other) continue;
                    if (other == &body) continue; // Skip self
                    
                    Vector<D> diff = other->position - body.position;
                    double dist_sq = diff.magnitude_squared();
                    
                    if (dist_sq < 1e-9) continue;
                    
                    double dist = std::sqrt(dist_sq);
                    double force_mag = G * body.mass * other->mass / (dist_sq * dist);
                    
                    forces[i] += diff.normalized() * force_mag;
                }
            }
            
            // Also handle direct calculations within own leaf
            for (const Body<D>* other : leaf->bodies) {
                if (other == &body) continue; // Skip self
                
                Vector<D> diff = other->position - body.position;
                double dist_sq = diff.magnitude_squared();
                
                if (dist_sq < 1e-9) continue;
                
                double dist = std::sqrt(dist_sq);
                double force_mag = G * body.mass * other->mass / (dist_sq * dist);
                
                forces[i] += diff.normalized() * force_mag;
            }
            
            // Evaluate local expansion
            if constexpr (D == 2) {
                std::complex<double> z = to_complex(body.position - leaf->center);
                std::complex<double> potential_gradient(0.0, 0.0);
                
                for (int p = 1; p <= order; ++p) {
                    std::complex<double> z_power = pow(z, p-1);
                    potential_gradient += leaf->local.coeff[p] * static_cast<double>(p) * z_power;
                }
                
                forces[i][0] += -potential_gradient.real() * body.mass;
                forces[i][1] += -potential_gradient.imag() * body.mass;
            } else {
                // 3D implementation (simplified)
                double potential = leaf->local.coeff[0].real();
                Vector<D> force_direction = leaf->center - body.position;
                double dist = force_direction.magnitude();
                
                if (dist > 1e-9) {
                    double force_mag = potential * body.mass / (dist * dist);
                    forces[i] += force_direction.normalized() * force_mag;
                }
            }
        }
    }
    
    return forces;
}

// FMM ParlayLib implementation
template <int D>
parlay::sequence<Vector<D>> fmm_parlay_n_body(const parlay::sequence<Body<D>>& bodies, 
                                             int max_bodies_per_leaf, 
                                             int max_level, 
                                             int order) {
    // Use higher order for better accuracy (minimum 8 for good results)
    order = std::max(order, 8);
    
    // Convert to std::vector for FMM initialization
    std::vector<Body<D>> std_bodies(bodies.begin(), bodies.end());
    
    // Create FMM instance
    FMM<D> fmm(std_bodies, max_bodies_per_leaf, max_level, order);
    
    // Get forces using sequential implementation first
    std::vector<Vector<D>> std_forces = fmm.calculate_forces(std_bodies);
    
    // Convert to parlay::sequence
    parlay::sequence<Vector<D>> forces(std_forces.begin(), std_forces.end());
    
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
