#ifndef FMM_TPP
#define FMM_TPP

#include "fmm.h"
#include "utils.h" // Helper functions moved to utils.h
#include <cmath>
#include <complex>
#include <algorithm>

// FMMNode implementation
template <int D>
FMMNode<D>::FMMNode(const Vector<D>& c, double hs, FMMNode* p, int lvl)
    : center(c), half_size(hs), parent(p), level(lvl) {
    // Initialize children to null
    for (auto& child : children) {
        child = nullptr;
    }
}

template <int D>
bool FMMNode<D>::is_leaf() const {
    // Check if all children are null
    for (const auto& child : children) {
        if (child) return false;
    }
    return true;
}

template <int D>
void FMMNode<D>::insert(Body<D>* body, int max_bodies, int max_level) {
    // If leaf node with capacity, just add the body
    if (is_leaf() && (bodies.size() < static_cast<size_t>(max_bodies) || level >= max_level)) {
        bodies.push_back(body);
        return;
    }

    // Need to subdivide
    if (is_leaf() && !bodies.empty()) {
        // Create children nodes and redistribute bodies
        for (int i = 0; i < (1 << D); ++i) {
            Vector<D> child_center = center;
            for (int d = 0; d < D; ++d) {
                child_center[d] += ((i & (1 << d)) ? 0.5 : -0.5) * half_size;
            }
            children[i] = std::make_unique<FMMNode>(child_center, half_size * 0.5, this, level + 1);
        }

        // Move existing bodies to children
        std::vector<Body<D>*> existing_bodies = std::move(bodies);
        bodies.clear(); // Clear this node's bodies

        for (Body<D>* existing_body : existing_bodies) {
            // Determine which child to insert into
            int child_idx = 0;
            for (int d = 0; d < D; ++d) {
                if (existing_body->position[d] > center[d]) {
                    child_idx |= (1 << d);
                }
            }
            children[child_idx]->insert(existing_body, max_bodies, max_level);
        }
    }

    // Now insert the new body into appropriate child
    int child_idx = 0;
    for (int d = 0; d < D; ++d) {
        if (body->position[d] > center[d]) {
            child_idx |= (1 << d);
        }
    }

    children[child_idx]->insert(body, max_bodies, max_level);
}

// Compute multipole expansion with improved accuracy
template <int D>
void FMMNode<D>::compute_multipole(int order) {
    multipole.clear();
    
    if (is_leaf()) {
        // P2M: Particle to Multipole
        if constexpr (D == 2) {
            // 2D implementation with complex numbers
            // First sum up all masses
            double total_mass = 0.0;
            Vector<D> center_of_mass = Vector<D>();
            
            for (Body<D>* body : bodies) {
                total_mass += body->mass;
                center_of_mass += body->position * body->mass;
            }
            
            if (total_mass > 0) {
                center_of_mass = center_of_mass / total_mass;
            }
            
            // Store total mass and center of mass
            multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            if (order > 0) {
                // Store center of mass in first coefficient for improved accuracy
                std::complex<double> z_com = to_complex(center_of_mass - center);
                multipole.coeff[1] = z_com * total_mass;
            }
            
            // Higher order terms for improved accuracy
            for (Body<D>* body : bodies) {
                std::complex<double> z = to_complex(body->position - center);
                std::complex<double> z_power = z * z; // Start with z^2
                
                for (int p = 2; p <= order; ++p) {
                    // Use normalized coefficients for better numerical stability
                    multipole.coeff[p] += -body->mass * z_power / static_cast<double>(p);
                    z_power *= z; // Next power
                }
            }
        } else {
            // 3D implementation with improved monopole and dipole terms
            double total_mass = 0.0;
            Vector<D> center_of_mass = Vector<D>();
            
            for (Body<D>* body : bodies) {
                total_mass += body->mass;
                center_of_mass += body->position * body->mass;
            }
            
            if (total_mass > 0) {
                center_of_mass = center_of_mass / total_mass;
            }
            
            // Store mass in first coefficient
            multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            if (order > 0 && multipole.coeff.size() > 2) {
                // Store center of mass for improved accuracy
                multipole.coeff[1] = std::complex<double>(center_of_mass[0], center_of_mass[1]);
                multipole.coeff[2] = std::complex<double>(center_of_mass[2], 0.0);
            }
        }
    } else {
        // Rest of the M2M implementation
        for (const auto& child : children) {
            if (child) {
                child->compute_multipole(order);
                
                if constexpr (D == 2) {
                    // 2D implementation with complex numbers
                    std::complex<double> z0 = to_complex(child->center - center);
                    
                    multipole.coeff[0] += child->multipole.coeff[0]; // Mass term
                    
                    // Compute higher order terms
                    for (int p = 1; p <= order; ++p) {
                        // Sum contributions from child's multipole moments
                        multipole.coeff[p] += child->multipole.coeff[p];
                        
                        // Adjust for the shift in coordinates using binomial expansion
                        for (int k = 0; k < p; ++k) {
                            std::complex<double> term = 
                                child->multipole.coeff[k] * pow(-z0, p-k) * 
                                static_cast<double>(binomial(p, k));
                            multipole.coeff[p] += term;
                        }
                    }
                } else {
                    // 3D implementation with spherical harmonics goes here
                    multipole.coeff[0] += child->multipole.coeff[0]; // Mass term for demo
                }
            }
        }
    }
}

// Translate multipole to local with improved accuracy
template <int D>
void FMMNode<D>::translate_multipole_to_local(FMMNode<D>* source, int order) {
    if (!source) return;
    
    // Get source mass
    double src_mass = source->multipole.coeff[0].real();
    if (src_mass < 1e-10) return;
    
    if constexpr (D == 2) {
        // Get the vector from target to source
        std::complex<double> z0 = to_complex(source->center - center);
        double r = std::abs(z0);
        
        // Skip if centers are too close (numerical stability)
        if (r < 1e-10) return;
        
        // Extract center of mass from source if available
        std::complex<double> source_com = z0;
        if (order > 0 && std::abs(source->multipole.coeff[1]) > 1e-10) {
            source_com = source->multipole.coeff[1] / src_mass + z0;
        }
        
        // More stable M2L computation
        for (int p = 0; p <= order; ++p) {
            for (int q = 0; q <= order; ++q) {
                std::complex<double> term;
                if (q == 0) {
                    // Monopole term - use source mass at COM
                    if (p == 0) {
                        term = src_mass / r;
                    } else {
                        term = src_mass * std::pow(1.0 / source_com, p);
                    }
                } else if (p + q <= order) {
                    // Higher order terms with improved stability
                    term = source->multipole.coeff[q] * 
                          std::pow(1.0 / z0, p + q) *
                          static_cast<double>(binomial(p + q - 1, q - 1));
                }
                local.coeff[p] += term;
            }
        }
    } else {
        // 3D implementation with improved accuracy
        Vector<D> r_vec = source->center - center;
        double r = r_vec.magnitude();
        
        // Skip if too close for numerical stability
        if (r < 1e-10) return;
        
        // Extract center of mass if available
        Vector<D> source_com = source->center;
        if (source->multipole.coeff.size() > 2) {
            source_com[0] = source->multipole.coeff[1].real();
            source_com[1] = source->multipole.coeff[1].imag();
            source_com[2] = source->multipole.coeff[2].real();
        }
        
        // Vector from target to source COM
        Vector<D> r_com_vec = source_com - center;
        double r_com = r_com_vec.magnitude();
        
        // Store potential from monopole approximation
        if (r_com > 1e-10) {
            local.coeff[0] += std::complex<double>(src_mass / r_com, 0.0);
            
            // Store direction for force calculation
            if (local.coeff.size() > 2) {
                local.coeff[1] = std::complex<double>(r_com_vec[0], r_com_vec[1]);
                local.coeff[2] = std::complex<double>(r_com_vec[2], r_com);
            }
        }
    }
}

template <int D>
void FMMNode<D>::translate_local_to_children(int order) {
    for (auto& child : children) {
        if (child) {
            if constexpr (D == 2) {
                std::complex<double> z0 = to_complex(center - child->center);
                
                // Transfer local expansions from parent to child
                for (int p = 0; p <= order; ++p) {
                    child->local.coeff[p] += local.coeff[p];
                    
                    for (int q = p + 1; q <= order; ++q) {
                        std::complex<double> term = local.coeff[q] * 
                                                   pow(z0, q - p) *
                                                   static_cast<double>(binomial(q, p));
                        child->local.coeff[p] += term;
                    }
                }
            } else {
                // 3D implementation with spherical harmonics
                // Fix division of complex by int by using static_cast
                child->local.coeff[0] += local.coeff[0] / static_cast<double>((1 << D));
            }
        }
    }
}

template <int D>
void FMMNode<D>::compute_direct_forces(std::vector<Vector<D>>& forces, 
                                       const std::vector<Body<D>>& all_bodies) {
    if (is_leaf() && !bodies.empty()) {
        // Direct calculation between bodies in this leaf and nearby leaves
        for (Body<D>* body_ptr : bodies) {
            size_t body_idx = body_ptr - &all_bodies[0]; // Find index of body
            
            // Compute forces with bodies in this node
            for (Body<D>* other_ptr : bodies) {
                if (body_ptr == other_ptr) continue; // Skip self-interaction
                
                Vector<D> diff = other_ptr->position - body_ptr->position;
                double dist_sq = diff.magnitude_squared();
                
                if (dist_sq < 1e-9) continue; // Avoid division by zero
                
                double dist = std::sqrt(dist_sq);
                double force_mag = G * body_ptr->mass * other_ptr->mass / (dist_sq * dist);
                
                forces[body_idx] += diff.normalized() * force_mag;
            }
            
            // Compute forces with bodies in neighbor nodes
            for (FMMNode<D>* neighbor : neighbor_list) {
                if (!neighbor->is_leaf()) continue;
                
                for (Body<D>* other_ptr : neighbor->bodies) {
                    Vector<D> diff = other_ptr->position - body_ptr->position;
                    double dist_sq = diff.magnitude_squared();
                    
                    if (dist_sq < 1e-9) continue; // Avoid division by zero
                    
                    double dist = std::sqrt(dist_sq);
                    double force_mag = G * body_ptr->mass * other_ptr->mass / (dist_sq * dist);
                    
                    forces[body_idx] += diff.normalized() * force_mag;
                }
            }
        }
    }
}

// Evaluate local expansion with improved accuracy
template <int D>
void FMMNode<D>::evaluate_local_expansion(std::vector<Vector<D>>& forces,
                                          const std::vector<Body<D>>& all_bodies,
                                          int order) {
    if (!is_leaf() || bodies.empty()) return;
    
    for (Body<D>* body_ptr : bodies) {
        size_t body_idx = body_ptr - &all_bodies[0]; // Find index of body
        
        if constexpr (D == 2) {
            std::complex<double> z = to_complex(body_ptr->position - center);
            std::complex<double> potential(0.0, 0.0);
            std::complex<double> potential_gradient(0.0, 0.0);
            
            // More accurate gradient calculation
            for (int p = 1; p <= order; ++p) {
                // Skip small coefficients
                if (std::abs(local.coeff[p]) < 1e-15) continue;
                
                std::complex<double> z_power;
                if (p == 1) {
                    z_power = std::complex<double>(1.0, 0.0);
                } else {
                    z_power = pow(z, p-1);
                }
                
                potential_gradient += local.coeff[p] * static_cast<double>(p) * z_power;
                potential += local.coeff[p] * pow(z, p);
            }
            
            // Force is negative gradient of potential
            Vector<D> force;
            force[0] = -potential_gradient.real() * body_ptr->mass;
            force[1] = -potential_gradient.imag() * body_ptr->mass;
            
            forces[body_idx] += force;
        } else {
            // 3D implementation
            double potential = local.coeff[0].real();
            Vector<D> r_vec;
            double r = 1.0;
            
            // Use stored direction and distance if available
            if (local.coeff.size() > 2) {
                r_vec[0] = local.coeff[1].real();
                r_vec[1] = local.coeff[1].imag();
                r_vec[2] = local.coeff[2].real();
                r = local.coeff[2].imag();
                
                if (r > 1e-10) {
                    // Normalize to unit vector
                    r_vec = r_vec / r;
                    
                    // Force magnitude
                    double force_mag = G * body_ptr->mass * potential / r;
                    
                    // Apply force toward the center of mass
                    forces[body_idx] -= r_vec * force_mag;
                }
            }
        }
    }
}

// FMM class implementation
template <int D>
FMM<D>::FMM(const std::vector<Body<D>>& bodies, int max_bodies, int max_lvl, int p)
    : max_bodies_per_leaf(max_bodies), max_level(max_lvl), order(p) {
    build_tree(bodies);
    build_interaction_lists();
}

template <int D>
void FMM<D>::build_tree(const std::vector<Body<D>>& bodies) {
    if (bodies.empty()) return;
    
    // Find bounding box
    Vector<D> min_pos = bodies[0].position;
    Vector<D> max_pos = bodies[0].position;
    
    for (const auto& body : bodies) {
        for (int d = 0; d < D; ++d) {
            min_pos[d] = std::min(min_pos[d], body.position[d]);
            max_pos[d] = std::max(max_pos[d], body.position[d]);
        }
    }
    
    // Calculate center and half size
    Vector<D> center;
    double max_half_size = 0;
    
    for (int d = 0; d < D; ++d) {
        center[d] = (min_pos[d] + max_pos[d]) / 2.0;
        max_half_size = std::max(max_half_size, std::abs(max_pos[d] - min_pos[d]) / 2.0);
    }
    
    // Add padding
    max_half_size *= 1.1;
    
    // Create body pointers for easier manipulation
    std::vector<Body<D>*> body_ptrs;
    body_ptrs.reserve(bodies.size());
    
    for (size_t i = 0; i < bodies.size(); ++i) {
        body_ptrs.push_back(const_cast<Body<D>*>(&bodies[i]));
    }
    
    // Build tree recursively
    root = build_tree_recursive(center, max_half_size, body_ptrs, nullptr, 0);
}

template <int D>
std::unique_ptr<FMMNode<D>> FMM<D>::build_tree_recursive(
    const Vector<D>& center, 
    double half_size,
    std::vector<Body<D>*>& bodies_subset,
    FMMNode<D>* parent,
    int level) {
    
    auto node = std::make_unique<FMMNode<D>>(center, half_size, parent, level);
    
    // If few enough bodies or max level reached, make leaf node
    if (bodies_subset.size() <= static_cast<size_t>(max_bodies_per_leaf) || level >= max_level) {
        node->bodies = bodies_subset;
        return node;
    }
    
    // Sort bodies into children
    std::array<std::vector<Body<D>*>, 1 << D> child_bodies;
    
    for (Body<D>* body : bodies_subset) {
        int child_idx = 0;
        for (int d = 0; d < D; ++d) {
            if (body->position[d] > center[d]) {
                child_idx |= (1 << d);
            }
        }
        child_bodies[child_idx].push_back(body);
    }
    
    // Create child nodes
    for (int i = 0; i < (1 << D); ++i) {
        if (!child_bodies[i].empty()) {
            Vector<D> child_center = center;
            for (int d = 0; d < D; ++d) {
                child_center[d] += ((i & (1 << d)) ? 0.5 : -0.5) * half_size;
            }
            node->children[i] = build_tree_recursive(
                child_center, 
                half_size * 0.5, 
                child_bodies[i], 
                node.get(), 
                level + 1
            );
        }
    }
    
    return node;
}

// Build interaction lists with improved accuracy criterion
template <int D>
void FMM<D>::build_interaction_lists_recursive(FMMNode<D>* node) {
    // Clear existing lists
    node->interaction_list.clear();
    node->neighbor_list.clear();
    
    // Use more conservative well-separateness criterion
    if (node->is_leaf()) {
        std::function<void(FMMNode<D>*)> process_node;
        process_node = [&](FMMNode<D>* other) {
            if (!other || other == node) return;
            
            // Calculate distance between node centers
            Vector<D> dist_vec = other->center - node->center;
            double dist = dist_vec.magnitude();
            
            // More conservative criterion for better accuracy
            double size_sum = node->half_size + other->half_size;
            
            // Use smaller threshold for more accurate approximation
            if (dist > 2.5 * size_sum) {
                // Well-separated: use multipole approximation
                node->interaction_list.push_back(other);
            } else if (other->is_leaf()) {
                // Close leaf: use direct calculation
                node->neighbor_list.push_back(other);
            } else {
                // Close internal node: recurse to children
                for (auto& child : other->children) {
                    if (child) process_node(child.get());
                }
            }
        };
        
        // Start from root
        if (root && root.get() != node) {
            process_node(root.get());
        }
    } else {
        // For internal nodes, process children recursively
        for (auto& child : node->children) {
            if (child) build_interaction_lists_recursive(child.get());
        }
    }
}

template <int D>
void FMM<D>::build_interaction_lists() {
    if (root) {
        build_interaction_lists_recursive(root.get());
    }
}

template <int D>
void FMM<D>::execute_fmm() {
    upward_pass();
    interaction_pass();
}

template <int D>
void FMM<D>::upward_pass() {
    if (!root) return;
    
    // Post-order traversal to compute multipoles bottom-up
    // Fix recursive lambda by capturing itself
    std::function<void(FMMNode<D>*)> compute;
    compute = [this, &compute](FMMNode<D>* node) {
        if (!node) return;
        
        // First, compute for all children
        for (auto& child : node->children) {
            if (child) compute(child.get());
        }
        
        // Then compute for this node
        node->compute_multipole(this->order);
    };
    
    compute(root.get());
}

template <int D>
void FMM<D>::interaction_pass() {
    if (!root) return;
    
    // Process all nodes in the tree
    // Fix recursive lambda by capturing itself
    std::function<void(FMMNode<D>*)> process;
    process = [this, &process](FMMNode<D>* node) {
        if (!node) return;
        
        // Process M2L for this node's interaction list
        for (FMMNode<D>* interaction : node->interaction_list) {
            node->translate_multipole_to_local(interaction, this->order);
        }
        
        // Recurse to children
        for (auto& child : node->children) {
            if (child) process(child.get());
        }
    };
    
    process(root.get());
}

template <int D>
void FMM<D>::downward_pass(std::vector<Vector<D>>& forces, const std::vector<Body<D>>& bodies) {
    if (!root) return;
    
    // First, translate local expansions from parents to children (top-down)
    // Fix recursive lambda by capturing itself
    std::function<void(FMMNode<D>*)> translate_local;
    translate_local = [this, &translate_local](FMMNode<D>* node) {
        if (!node) return;
        
        // Translate to children
        if (!node->is_leaf()) {
            node->translate_local_to_children(this->order);
            
            // Continue down the tree
            for (auto& child : node->children) {
                if (child) translate_local(child.get());
            }
        }
    };
    
    translate_local(root.get());
    
    // Then, compute direct forces and evaluate local expansions at leaf nodes
    std::function<void(FMMNode<D>*)> compute_forces = [&](FMMNode<D>* node) {
        if (!node) return;
        
        if (node->is_leaf()) {
            // For leaf nodes, compute direct forces and evaluate local expansion
            node->compute_direct_forces(forces, bodies);
            node->evaluate_local_expansion(forces, bodies, this->order);
        } else {
            // Recurse to children
            for (auto& child : node->children) {
                if (child) compute_forces(child.get());
            }
        }
    };
    
    compute_forces(root.get());
}

// Calculate force on a single body with improved accuracy
template <int D>
Vector<D> FMM<D>::calculate_accurate_force(const Body<D>& body) {
    if (!root) return Vector<D>();
    
    // Use Barnes-Hut style traversal with improved accuracy
    std::function<Vector<D>(FMMNode<D>*, const Body<D>&)> calc_force;
    calc_force = [&calc_force](FMMNode<D>* node, const Body<D>& body) -> Vector<D> {
        if (!node) return Vector<D>();
        
        // Vector from body to node center
        Vector<D> diff = node->center - body.position;
        double dist_sq = diff.magnitude_squared();
        
        // To avoid division by zero
        if (dist_sq < 1e-10) {
            // If this is a non-leaf node, recurse to children
            if (!node->is_leaf()) {
                Vector<D> force;
                for (auto& child : node->children) {
                    if (child) {
                        force += calc_force(child.get(), body);
                    }
                }
                return force;
            } else {
                // Leaf node with very close points - compute direct forces
                Vector<D> force;
                for (Body<D>* other : node->bodies) {
                    if (other == &body) continue; // Skip self
                    
                    Vector<D> r = other->position - body.position;
                    double r_sq = r.magnitude_squared();
                    if (r_sq < 1e-10) continue;
                    
                    double r_mag = std::sqrt(r_sq);
                    double force_mag = G * body.mass * other->mass / (r_sq * r_mag);
                    force += r.normalized() * force_mag;
                }
                return force;
            }
        }
        
        // Multipole acceptance criterion - more conservative
        double dist = std::sqrt(dist_sq);
        if (node->half_size / dist < 0.3) { // More conservative
            // Get total mass
            double total_mass = node->multipole.coeff[0].real();
            
            // Try using center of mass for better accuracy
            Vector<D> com_pos = node->center;
            if constexpr (D == 2) {
                if (node->multipole.coeff.size() > 1) {
                    // Extract center of mass from first multipole term
                    std::complex<double> z_com = node->multipole.coeff[1] / total_mass;
                    com_pos[0] = node->center[0] + z_com.real();
                    com_pos[1] = node->center[1] + z_com.imag();
                }
            } else {
                if (node->multipole.coeff.size() > 2) {
                    com_pos[0] = node->multipole.coeff[1].real();
                    com_pos[1] = node->multipole.coeff[1].imag();
                    com_pos[2] = node->multipole.coeff[2].real();
                }
            }
            
            // Calculate force using center of mass
            Vector<D> r = com_pos - body.position;
            double r_sq = r.magnitude_squared();
            if (r_sq < 1e-10) return Vector<D>();
            
            double r_mag = std::sqrt(r_sq);
            double force_mag = G * body.mass * total_mass / (r_sq * r_mag);
            return r.normalized() * force_mag;
        } else {
            // Recurse to children
            if (node->is_leaf()) {
                // Direct calculation for leaf node
                Vector<D> force;
                for (Body<D>* other : node->bodies) {
                    if (other == &body) continue; // Skip self
                    
                    Vector<D> r = other->position - body.position;
                    double r_sq = r.magnitude_squared();
                    if (r_sq < 1e-10) continue;
                    
                    double r_mag = std::sqrt(r_sq);
                    double force_mag = G * body.mass * other->mass / (r_sq * r_mag);
                    force += r.normalized() * force_mag;
                }
                return force;
            } else {
                // Non-leaf, check children
                Vector<D> force;
                for (auto& child : node->children) {
                    if (child) {
                        force += calc_force(child.get(), body);
                    }
                }
                return force;
            }
        }
    };
    
    return calc_force(root.get(), body);
}

// Calculate forces with improved accuracy
template <int D>
std::vector<Vector<D>> FMM<D>::calculate_forces(const std::vector<Body<D>>& bodies) {
    std::vector<Vector<D>> forces(bodies.size(), Vector<D>());
    
    if (!root || bodies.empty()) {
        return forces;
    }
    
    // Use the improved accuracy calculation method
    for (size_t i = 0; i < bodies.size(); ++i) {
        forces[i] = calculate_accurate_force(bodies[i]);
    }
    
    return forces;
}

#endif // FMM_TPP
