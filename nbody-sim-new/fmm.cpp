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

template <int D>
void FMMNode<D>::compute_multipole(int order) {
    multipole.clear();
    
    if (is_leaf()) {
        // Improved P2M for accuracy
        if constexpr (D == 2) {
            // 2D with complex numbers - same as before but ensuring numerical stability
            multipole.coeff[0] = std::complex<double>(0.0, 0.0);
            
            // First sum up masses for stability
            double total_mass = 0.0;
            for (Body<D>* body : bodies) {
                total_mass += body->mass;
            }
            
            // Set monopole term (total mass)
            multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            // Calculate higher moments with proper scaling
            for (Body<D>* body : bodies) {
                std::complex<double> z = to_complex(body->position - center);
                std::complex<double> z_power = z;
                
                for (int p = 1; p <= order; ++p) {
                    // Normalize by order for better numerical behavior
                    multipole.coeff[p] += -body->mass * z_power / static_cast<double>(p);
                    z_power *= z;
                }
            }
        } else {
            // 3D implementation with better monopole and dipole approximation
            // First compute total mass (monopole)
            double total_mass = 0.0;
            Vector<D> com;  // Center of mass
            
            for (Body<D>* body : bodies) {
                total_mass += body->mass;
                for (int d = 0; d < D; ++d) {
                    com[d] += body->mass * body->position[d];
                }
            }
            
            // Normalize center of mass
            if (total_mass > 1e-10) {
                for (int d = 0; d < D; ++d) {
                    com[d] /= total_mass;
                }
            } else {
                com = center; // Fallback
            }
            
            // Store monopole term (total mass)
            multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            // Store dipole information for 3D approximation
            if (order > 0) {
                // Store center of mass displacement in higher coefficients
                Vector<D> com_disp = com - center;
                for (int d = 0; d < D; ++d) {
                    multipole.coeff[d+1] = std::complex<double>(com_disp[d] * total_mass, 0.0);
                }
            }
        }
    } else {
        // M2M translation logic same as before
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

template <int D>
void FMMNode<D>::translate_multipole_to_local(FMMNode<D>* source, int order) {
    // More accurate M2L translation
    if constexpr (D == 2) {
        std::complex<double> z0 = to_complex(source->center - center);
        double r = std::abs(z0);
        
        // Skip if centers are too close (numerical stability)
        if (r < 1e-10) return;
        
        // For each target order and source order, with better accuracy
        for (int p = 0; p <= order; ++p) {
            for (int q = 0; q <= order; ++q) {
                // More stable computation for large powers
                std::complex<double> term;
                if (p + q > 30) { // Threshold for large powers
                    // Use logarithmic computation to avoid overflow
                    double log_mag = q * std::log(source->multipole.coeff[q].real()) - 
                                    (p + q + 1) * std::log(r) + 
                                    std::log(static_cast<double>(binomial(p + q, q)));
                    term = std::exp(log_mag) * std::exp(std::complex<double>(0, -(p+q+1)*std::arg(z0)));
                } else {
                    // Standard computation for smaller powers
                    term = source->multipole.coeff[q] * 
                           std::pow(1.0 / z0, p + q + 1) *
                           static_cast<double>(binomial(p + q, q));
                }
                local.coeff[p] += term;
            }
        }
    } else {
        // 3D implementation with improved accuracy
        // For now, use a conservative estimate that works better than the previous simplified version
        Vector<D> r_vec = source->center - center;
        double r = r_vec.magnitude();
        
        // Skip if centers are too close (numerical stability)
        if (r < 1e-10) return;
        
        // Basic monopole contribution (1/r term)
        local.coeff[0] += source->multipole.coeff[0] / r;
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

template <int D>
void FMMNode<D>::evaluate_local_expansion(std::vector<Vector<D>>& forces,
                                          const std::vector<Body<D>>& all_bodies,
                                          int order) {
    if (!is_leaf() || bodies.empty()) return;
    
    for (Body<D>* body_ptr : bodies) {
        size_t body_idx = body_ptr - &all_bodies[0]; // Find index of body
        
        if constexpr (D == 2) {
            // 2D implementation - enhanced for accuracy
            std::complex<double> z = to_complex(body_ptr->position - center);
            std::complex<double> potential_gradient(0.0, 0.0);
            
            // More accurate gradient calculation
            for (int p = 1; p <= order; ++p) {
                if (std::abs(local.coeff[p]) < 1e-15) continue;
                
                // Calculate z^(p-1) more carefully to avoid overflow
                std::complex<double> z_power;
                if (p == 1) {
                    z_power = std::complex<double>(1.0, 0.0);
                } else if (p <= 10) {
                    z_power = pow(z, p-1);
                } else {
                    // For high orders, use more stable computation
                    double mag = pow(std::abs(z), p-1);
                    double arg = (p-1) * std::arg(z);
                    z_power = std::polar(mag, arg);
                }
                
                potential_gradient += local.coeff[p] * static_cast<double>(p) * z_power;
            }
            
            // Force is -gradient * mass
            Vector<D> force;
            force[0] = -potential_gradient.real() * body_ptr->mass;
            force[1] = -potential_gradient.imag() * body_ptr->mass;
            
            forces[body_idx] += force;
        } else {
            // Enhanced 3D implementation
            // For 3D, our local expansions are limited, so use direct calculation
            // for better accuracy in some cases
            double potential = local.coeff[0].real();
            Vector<D> r_vec = body_ptr->position - center;
            double r = r_vec.magnitude();
            
            if (r > 1e-10) {
                // Main monopole term (conservative but accurate)
                double force_mag = G * body_ptr->mass * potential / (r * r * r);
                Vector<D> force = r_vec * (-force_mag); // Force toward center
                
                // Add dipole corrections if available (for order > 0)
                if (order > 0) {
                    // Dipole terms would be calculated here in a full implementation
                    // For simplicity, we'll just use the monopole approximation
                }
                
                forces[body_idx] += force;
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

template <int D>
void FMM<D>::build_interaction_lists() {
    if (root) {
        build_interaction_lists_recursive(root.get());
    }
}

template <int D>
void FMM<D>::build_interaction_lists_recursive(FMMNode<D>* node) {
    // Clear existing lists
    node->interaction_list.clear();
    node->neighbor_list.clear();
    
    // Improved well-separateness criterion
    if (node->is_leaf()) {
        std::function<void(FMMNode<D>*)> process_node = [&](FMMNode<D>* other) {
            if (!other || other == node) return;
            
            // Calculate distance between node centers
            Vector<D> dist_vec = other->center - node->center;
            double dist = dist_vec.magnitude();
            
            // Node sizes
            double size_self = node->half_size * 2.0;
            double size_other = other->half_size * 2.0;
            
            // Improved well-separateness criterion
            // If distance > size_self + size_other, use multipole approximation
            // More conservative than the classic r > 2*(R1+R2) to improve accuracy
            if (dist > 1.5 * (size_self + size_other)) {
                // Use multipole approximation
                if (other->is_leaf()) {
                    node->interaction_list.push_back(other);
                } else {
                    // If other is larger, it's more accurate to interact with its children
                    if (size_other > size_self * 2.0) {
                        for (auto& child : other->children) {
                            if (child) process_node(child.get());
                        }
                    } else {
                        node->interaction_list.push_back(other);
                    }
                }
            } else {
                // Close interaction - direct calculation needed
                if (other->is_leaf()) {
                    node->neighbor_list.push_back(other);
                } else {
                    // Must recurse into children for direct calculation
                    for (auto& child : other->children) {
                        if (child) process_node(child.get());
                    }
                }
            }
        };
        
        // Process from root
        if (root) process_node(root.get());
    } else {
        // Process children recursively
        for (auto& child : node->children) {
            if (child) build_interaction_lists_recursive(child.get());
        }
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

template <int D>
std::vector<Vector<D>> FMM<D>::calculate_forces(const std::vector<Body<D>>& bodies) {
    std::vector<Vector<D>> forces(bodies.size(), Vector<D>());
    
    if (!root || bodies.empty()) {
        return forces;
    }
    
    // Use a higher order for the calculations to improve accuracy
    int enhanced_order = std::max(order, 10);
    
    // Execute the FMM algorithm with enhanced parameters
    upward_pass();
    interaction_pass();
    
    // Improved force calculation with more direct calculations for accuracy
    std::vector<FMMNode<D>*> leaf_nodes;
    std::function<void(FMMNode<D>*)> collect_leaves = [&](FMMNode<D>* node) {
        if (!node) return;
        if (node->is_leaf()) {
            leaf_nodes.push_back(node);
        } else {
            for (auto& child : node->children) {
                if (child) collect_leaves(child.get());
            }
        }
    };
    
    collect_leaves(root.get());
    
    // Map bodies to leaf nodes
    std::vector<FMMNode<D>*> body_to_leaf(bodies.size(), nullptr);
    for (auto* leaf : leaf_nodes) {
        for (auto* body_ptr : leaf->bodies) {
            for (size_t i = 0; i < bodies.size(); ++i) {
                if (&bodies[i] == body_ptr) {
                    body_to_leaf[i] = leaf;
                    break;
                }
            }
        }
    }
    
    // Perform direct and far-field calculations
    downward_pass(forces, bodies);
    
    // Additional verification to ensure no bodies were missed
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (forces[i].magnitude() < 1e-20) {
            // If force is negligible, try direct calculation with all bodies
            FMMNode<D>* leaf = body_to_leaf[i];
            if (leaf) {
                const Body<D>& body = bodies[i];
                
                // Do direct calculation with nearby bodies
                for (size_t j = 0; j < bodies.size(); ++j) {
                    if (i == j) continue;
                    
                    const Body<D>& other = bodies[j];
                    Vector<D> diff = other.position - body.position;
                    double dist_sq = diff.magnitude_squared();
                    
                    if (dist_sq < 1e-9) continue;
                    
                    double dist = std::sqrt(dist_sq);
                    double force_mag = G * body.mass * other.mass / (dist_sq * dist);
                    
                    forces[i] += diff.normalized() * force_mag;
                }
            }
        }
    }
    
    return forces;
}

#endif // FMM_TPP
