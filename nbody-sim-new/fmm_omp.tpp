#ifndef FMM_OMP_TPP
#define FMM_OMP_TPP

#include "fmm_omp.h"
#include <algorithm>
#include <vector>
#include <functional>

template <int D>
FMM_OMP<D>::FMM_OMP(const std::vector<Body<D>>& bodies, 
                   int max_bodies, 
                   int max_lvl, 
                   int p)
    : max_bodies_per_leaf(max_bodies), max_level(max_lvl), order(p)
{
    // Create a regular FMM object to build the tree and interaction lists
    FMM<D> fmm_builder(bodies, max_bodies, max_lvl, p);
    
    // Extract the root
    root = std::move(fmm_builder.root);
    
    // Build level lists for parallel processing
    build_level_lists();
    
    // Rebuild interaction lists with more conservative criteria
    build_interaction_lists();
}

template <int D>
void FMM_OMP<D>::build_level_lists() {
    if (!root) return;
    
    // Find maximum depth of the tree
    int max_depth = 0;
    
    std::function<void(FMMNode<D>*, int)> find_depth;
    find_depth = [&](FMMNode<D>* node, int depth) {
        if (!node) return;
        
        max_depth = std::max(max_depth, depth);
        
        for (auto& child : node->children) {
            if (child) find_depth(child.get(), depth + 1);
        }
    };
    
    find_depth(root.get(), 0);
    
    // Initialize level lists
    nodes_by_level.resize(max_depth + 1);
    
    // Populate level lists
    std::function<void(FMMNode<D>*, int)> collect_nodes;
    collect_nodes = [&](FMMNode<D>* node, int depth) {
        if (!node) return;
        
        // Add node to its level list
        nodes_by_level[depth].push_back(node);
        
        // Process children
        for (auto& child : node->children) {
            if (child) collect_nodes(child.get(), depth + 1);
        }
    };
    
    collect_nodes(root.get(), 0);
}

template <int D>
void FMM_OMP<D>::build_interaction_lists() {
    if (!root) return;
    
    // Use more conservative multipole acceptance criterion
    std::function<void(FMMNode<D>*)> process;
    process = [&](FMMNode<D>* node) {
        if (!node) return;
        
        // Clear existing lists
        node->interaction_list.clear();
        node->neighbor_list.clear();
        
        // Process for leaf nodes
        if (node->is_leaf()) {
            std::function<void(FMMNode<D>*)> check_node;
            check_node = [&](FMMNode<D>* other) {
                if (!other || other == node) return;
                
                // Calculate distance between node centers
                Vector<D> dist_vec = other->center - node->center;
                double dist = dist_vec.magnitude();
                
                // More conservative criterion for well-separateness: 0.3 instead of 0.5
                double size_sum = node->half_size + other->half_size;
                
                // More conservative criterion: at least 2.5 times the sum of half-sizes
                if (dist > 2.5 * size_sum || node->half_size / dist < 0.3) {
                    // Well-separated: use multipole
                    node->interaction_list.push_back(other);
                } else if (other->is_leaf()) {
                    // Close leaf: direct calculation
                    node->neighbor_list.push_back(other);
                } else {
                    // Close internal node: check children
                    for (auto& child : other->children) {
                        if (child) check_node(child.get());
                    }
                }
            };
            
            // Start from root
            if (root && root.get() != node) {
                check_node(root.get());
            }
        }
        
        // Process children recursively
        for (auto& child : node->children) {
            if (child) process(child.get());
        }
    };
    
    process(root.get());
}

// P2M phase optimized with OpenMP
template <int D>
void FMM_OMP<D>::p2m_phase() {
    if (nodes_by_level.empty()) return;
    
    // Get leaf nodes (last level might not all be leaves in adaptive tree)
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (size_t level = 0; level < nodes_by_level.size(); ++level) {
        for (FMMNode<D>* node : nodes_by_level[level]) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    // Parallelize P2M computation for all leaf nodes with improved accuracy
    #pragma omp parallel for
    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        FMMNode<D>* node = leaf_nodes[i];
        
        // Clear multipole
        node->multipole.clear();
        
        if constexpr (D == 2) {
            // Improved 2D implementation with enhanced stability
            double total_mass = 0.0;
            Vector<D> center_of_mass = Vector<D>();
            
            for (Body<D>* body : node->bodies) {
                total_mass += body->mass;
                center_of_mass += body->position * body->mass;
            }
            
            // More robust handling of empty or near-empty nodes
            if (total_mass > 1e-14) {
                center_of_mass = center_of_mass / total_mass;
            } else {
                center_of_mass = node->center;
                total_mass = 1e-14; // Avoid division by zero later
            }
            
            // Store total mass
            node->multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            // Store center of mass for improved accuracy
            if (order > 0) {
                std::complex<double> z_com = to_complex(center_of_mass - node->center);
                node->multipole.coeff[1] = z_com * total_mass;
            }
            
            // Compute higher-order terms with better numerical stability
            // Use center of mass as expansion center for better convergence
            for (Body<D>* body : node->bodies) {
                std::complex<double> z = to_complex(body->position - center_of_mass);
                std::complex<double> z_power = z; // Start with z^1
                
                for (int p = 2; p <= order; ++p) {
                    z_power *= z;  // Calculate z^p
                    // Use factorial from utils.h with explicit int cast
                    double norm_factor = 1.0 / static_cast<double>(::factorial(p));
                    node->multipole.coeff[p] += body->mass * z_power * norm_factor;
                }
            }
        } else {
            // 3D implementation with improved monopole and dipole terms
            double total_mass = 0.0;
            Vector<D> center_of_mass = Vector<D>();
            
            for (Body<D>* body : node->bodies) {
                total_mass += body->mass;
                center_of_mass += body->position * body->mass;
            }
            
            // More robust handling of empty or near-empty nodes
            if (total_mass > 1e-14) {
                center_of_mass = center_of_mass / total_mass;
            } else {
                center_of_mass = node->center;
                total_mass = 1e-14; // Avoid division by zero later
            }
            
            // Store mass in first coefficient
            node->multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            // Store center of mass for better accuracy in higher-order terms
            if (order > 0 && node->multipole.coeff.size() > 2) {
                node->multipole.coeff[1] = std::complex<double>(center_of_mass[0], center_of_mass[1]);
                node->multipole.coeff[2] = std::complex<double>(center_of_mass[2], 0.0);
            }
            
            // Add higher-order multipole terms if needed
            for (Body<D>* body : node->bodies) {
                // Use vector from COM to body for better convergence
                Vector<D> r_vec = body->position - center_of_mass;
                double r_squared = r_vec.magnitude_squared();
                
                if (r_squared > 1e-10 && order > 2) {
                    // Quadrupole moment contribution
                    for (int d1 = 0; d1 < D; ++d1) {
                        for (int d2 = d1; d2 < D; ++d2) {
                            int idx = 3 + d1 * D + d2; // Map to higher coefficients
                            if (idx < node->multipole.coeff.size()) {
                                // Calculate quadrupole term with better numerical stability
                                double quad_term = 0.5 * body->mass * 
                                    (3.0 * r_vec[d1] * r_vec[d2] - (d1 == d2 ? r_squared : 0.0));
                                
                                node->multipole.coeff[idx] += std::complex<double>(quad_term, 0.0);
                            }
                        }
                    }
                    
                    // Octupole moment for even higher accuracy when order > 3
                    if (order > 3) {
                        double r_cubed = r_squared * std::sqrt(r_squared);
                        if (r_cubed > 1e-14) {
                            for (int d1 = 0; d1 < D; ++d1) {
                                for (int d2 = 0; d2 < D; ++d2) {
                                    for (int d3 = d2; d3 < D; ++d3) {
                                        int idx = 12 + (d1 * D * D) + (d2 * D) + d3;
                                        if (idx < node->multipole.coeff.size()) {
                                            double oct_term = body->mass * r_vec[d1] * r_vec[d2] * r_vec[d3] / r_cubed;
                                            node->multipole.coeff[idx] += std::complex<double>(oct_term, 0.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// M2M phase optimized with OpenMP (level by level, bottom up)
template <int D>
void FMM_OMP<D>::m2m_phase() {
    if (nodes_by_level.empty()) return;
    
    // Process levels from bottom up, skip leaf level
    for (int level = static_cast<int>(nodes_by_level.size()) - 1; level > 0; --level) {
        std::vector<FMMNode<D>*>& nodes = nodes_by_level[level];
        
        // Get parent nodes that need M2M computation
        std::vector<FMMNode<D>*> parent_nodes;
        std::vector<bool> parent_processed(nodes_by_level[level-1].size(), false);
        
        for (FMMNode<D>* node : nodes) {
            if (node && node->parent) {
                size_t parent_idx = 0;
                while (parent_idx < nodes_by_level[level-1].size() && 
                       nodes_by_level[level-1][parent_idx] != node->parent) {
                    ++parent_idx;
                }
                
                if (parent_idx < nodes_by_level[level-1].size() && !parent_processed[parent_idx]) {
                    parent_nodes.push_back(node->parent);
                    parent_processed[parent_idx] = true;
                }
            }
        }
        
        // Parallelize M2M computation for all parent nodes at this level
        #pragma omp parallel for
        for (size_t i = 0; i < parent_nodes.size(); ++i) {
            FMMNode<D>* node = parent_nodes[i];
            
            // Clear multipole first
            node->multipole.clear();
            
            // Accumulate from all children
            for (auto& child : node->children) {
                if (child) {
                    // M2M: Child multipole to parent multipole
                    if constexpr (D == 2) {
                        // 2D complex-based translation
                        std::complex<double> z0 = to_complex(child->center - node->center);
                        
                        // Add mass term
                        node->multipole.coeff[0] += child->multipole.coeff[0];
                        
                        // Add higher order terms with binomial corrections
                        for (int p = 1; p <= order; ++p) {
                            node->multipole.coeff[p] += child->multipole.coeff[p];
                            
                            for (int k = 0; k < p; ++k) {
                                std::complex<double> term = 
                                    child->multipole.coeff[k] * pow(-z0, p-k) * 
                                    static_cast<double>(binomial(p, k));
                                node->multipole.coeff[p] += term;
                            }
                        }
                    } else {
                        // 3D implementation
                        // Add mass term
                        node->multipole.coeff[0] += child->multipole.coeff[0];
                        
                        // Add COM information
                        if (node->multipole.coeff.size() > 2 && child->multipole.coeff.size() > 2) {
                            double child_mass = child->multipole.coeff[0].real();
                            if (child_mass > 1e-10) {
                                double current_mass = node->multipole.coeff[0].real() - child_mass;
                                
                                if (current_mass > 1e-10) {
                                    // Update COM weighted by masses
                                    Vector<D> child_com, current_com, new_com;
                                    child_com[0] = child->multipole.coeff[1].real();
                                    child_com[1] = child->multipole.coeff[1].imag();
                                    child_com[2] = child->multipole.coeff[2].real();
                                    
                                    current_com[0] = node->multipole.coeff[1].real();
                                    current_com[1] = node->multipole.coeff[1].imag();
                                    current_com[2] = node->multipole.coeff[2].real();
                                    
                                    double new_mass = current_mass + child_mass;
                                    for (int d = 0; d < D; ++d) {
                                        new_com[d] = (current_com[d] * current_mass + 
                                                     child_com[d] * child_mass) / new_mass;
                                    }
                                    
                                    node->multipole.coeff[1] = std::complex<double>(new_com[0], new_com[1]);
                                    node->multipole.coeff[2] = std::complex<double>(new_com[2], 0.0);
                                } else {
                                    // If this was the first child, just copy its COM
                                    node->multipole.coeff[1] = child->multipole.coeff[1];
                                    node->multipole.coeff[2] = child->multipole.coeff[2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// M2L phase optimized with OpenMP
template <int D>
void FMM_OMP<D>::m2l_phase() {
    if (nodes_by_level.empty()) return;
    
    // Process all nodes that have interaction lists
    std::vector<FMMNode<D>*> nodes_with_interactions;
    
    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && !node->interaction_list.empty()) {
                nodes_with_interactions.push_back(node);
            }
        }
    }
    
    // Parallelize M2L computations
    #pragma omp parallel for
    for (size_t i = 0; i < nodes_with_interactions.size(); ++i) {
        FMMNode<D>* node = nodes_with_interactions[i];
        
        node->local.clear();
        
        for (FMMNode<D>* source : node->interaction_list) {
            if (!source) continue;
            
            double src_mass = source->multipole.coeff[0].real();
            if (src_mass < 1e-12) continue;  // Stricter threshold
            
            if constexpr (D == 2) {
                // Extract accurate center of mass
                Vector<D> source_com = source->center;
                if (order > 0 && std::abs(source->multipole.coeff[1]) > 1e-12) {
                    std::complex<double> z_com = source->multipole.coeff[1] / src_mass;
                    source_com[0] = source->center[0] + z_com.real();
                    source_com[1] = source->center[1] + z_com.imag();
                }
                
                // Improved vector calculation using COM
                Vector<D> r_vec = source_com - node->center;
                double r = r_vec.magnitude();
                
                if (r < 1e-10) continue;
                
                // Use the improved r_vec based on center of mass
                std::complex<double> z0 = to_complex(r_vec);
                
                // Enhanced and more stable M2L translation
                for (int p = 0; p <= order; ++p) {
                    // Use more accurate monopole approximation (p=0)
                    if (p == 0) {
                        // Direct monopole calculation
                        node->local.coeff[0] += std::complex<double>(src_mass / r, 0.0);
                        continue;
                    }
                    
                    for (int q = 0; q <= order-p; ++q) {
                        std::complex<double> term;
                        if (q == 0) {
                            // More stable monopole term
                            term = src_mass * std::pow(1.0 / std::max(r, 1e-10), p);
                        } else {
                            // Higher order with better numerical stability
                            double bin_coeff = binomial(p+q-1, q-1);
                            // Improved accuracy with direct division
                            std::complex<double> factor = source->multipole.coeff[q] / std::pow(z0, p+q);
                            term = bin_coeff * factor;
                        }
                        
                        // Add term if significant
                        if (std::abs(term) > 1e-14) {
                            node->local.coeff[p] += term;
                        }
                    }
                }
            } else {
                // 3D implementation with improved accuracy
                Vector<D> r_vec = source->center - node->center;
                double r = r_vec.magnitude();
                
                // Skip if too close for numerical stability
                if (r < 1e-10) continue;
                
                // Extract center of mass if available
                Vector<D> source_com = source->center;
                if (source->multipole.coeff.size() > 2) {
                    source_com[0] = source->multipole.coeff[1].real();
                    source_com[1] = source->multipole.coeff[1].imag();
                    source_com[2] = source->multipole.coeff[2].real();
                }
                
                // Vector from target to source COM for better accuracy
                Vector<D> r_com_vec = source_com - node->center;
                double r_com = r_com_vec.magnitude();
                
                // More robust distance handling
                if (r_com < 1e-10) continue;
                
                // Calculate normalized direction vector
                Vector<D> r_com_norm = r_com_vec / r_com;
                
                // Store potential from monopole approximation
                node->local.coeff[0] += std::complex<double>(src_mass / r_com, 0.0);
                
                // Store direction for force calculation
                if (node->local.coeff.size() > 2) {
                    node->local.coeff[1] = std::complex<double>(r_com_norm[0], r_com_norm[1]);
                    node->local.coeff[2] = std::complex<double>(r_com_norm[2], r_com); // Store distance in imag part
                }
                
                // Add quadrupole contribution if available
                if (order > 2) {
                    double r_com_sq = r_com * r_com;
                    double r_com_5 = r_com_sq * r_com_sq * r_com;
                    
                    for (int d1 = 0; d1 < D; ++d1) {
                        for (int d2 = d1; d2 < D; ++d2) {
                            int idx = 3 + d1 * D + d2;
                            if (idx < source->multipole.coeff.size()) {
                                double quad_term = source->multipole.coeff[idx].real();
                                
                                if (std::abs(quad_term) > 1e-12) {
                                    // Improved quadrupole contribution to potential
                                    double contrib = 0.5 * quad_term / r_com_5;
                                    
                                    // Direction-dependent correction for quadrupole
                                    double dir_factor = 5.0 * r_com_norm[d1] * r_com_norm[d2];
                                    if (d1 == d2) dir_factor -= 1.0;
                                    contrib *= dir_factor;
                                    
                                    // Store in higher coefficients for later use
                                    if (idx < node->local.coeff.size()) {
                                        node->local.coeff[idx] += std::complex<double>(contrib, 0.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// L2L phase optimized with OpenMP (level by level, top down)
template <int D>
void FMM_OMP<D>::l2l_phase() {
    if (nodes_by_level.empty()) return;
    
    // Process levels from top down, skip root level
    for (size_t level = 1; level < nodes_by_level.size(); ++level) {
        std::vector<FMMNode<D>*>& nodes = nodes_by_level[level];
        
        // Parallelize L2L translation for all nodes at this level
        #pragma omp parallel for
        for (size_t i = 0; i < nodes.size(); ++i) {
            FMMNode<D>* node = nodes[i];
            if (node && node->parent) {
                // L2L: Parent local to child local
                if constexpr (D == 2) {
                    // 2D translation with complex arithmetic
                    std::complex<double> z0 = to_complex(node->parent->center - node->center);
                    
                    // Transfer local expansions from parent to child
                    for (int p = 0; p <= order; ++p) {
                        node->local.coeff[p] += node->parent->local.coeff[p];
                        
                        // Add binomial correction terms
                        for (int q = p + 1; q <= order; ++q) {
                            std::complex<double> term = node->parent->local.coeff[q] * 
                                                      pow(z0, q - p) *
                                                      static_cast<double>(binomial(q, p));
                            node->local.coeff[p] += term;
                        }
                    }
                } else {
                    // 3D implementation
                    node->local.coeff[0] += node->parent->local.coeff[0] / static_cast<double>((1 << D));
                    
                    // Copy direction and distance information for force calculation
                    if (node->local.coeff.size() > 2 && node->parent->local.coeff.size() > 2) {
                        node->local.coeff[1] = node->parent->local.coeff[1];
                        node->local.coeff[2] = node->parent->local.coeff[2];
                    }
                }
            }
        }
    }
}

// L2P and P2P phases optimized with OpenMP
template <int D>
void FMM_OMP<D>::l2p_phase(std::vector<Vector<D>>& forces, const std::vector<Body<D>>& bodies) {
    if (nodes_by_level.empty()) return;
    
    // Get leaf nodes
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    // Parallelize L2P computation for all leaf nodes
    #pragma omp parallel for
    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        FMMNode<D>* node = leaf_nodes[i];
        if (!node || node->bodies.empty()) continue;
        
        for (Body<D>* body_ptr : node->bodies) {
            size_t body_idx = 0;
            for (size_t j = 0; j < bodies.size(); ++j) {
                if (&bodies[j] == body_ptr) {
                    body_idx = j;
                    break;
                }
            }
            
            if constexpr (D == 2) {
                // Improved gradient calculation for 2D
                std::complex<double> z = to_complex(body_ptr->position - node->center);
                std::complex<double> potential(0.0, 0.0);
                std::complex<double> gradient(0.0, 0.0);
                
                // Calculate potential and its gradient more accurately
                for (int p = 0; p <= order; ++p) {
                    // Skip small coefficients for numerical stability
                    if (std::abs(node->local.coeff[p]) < 1e-14) continue;
                    
                    // Calculate z^(p-1) cautiously
                    std::complex<double> z_power = p > 0 ? std::pow(z, p-1) : std::complex<double>(1.0, 0.0);
                    
                    // Add to potential
                    potential += node->local.coeff[p] * std::pow(z, p);
                    
                    // Add to gradient (derivative of potential)
                    if (p > 0) {
                        gradient += static_cast<double>(p) * node->local.coeff[p] * z_power;
                    }
                }
                
                // Force is negative gradient of potential
                Vector<D> force;
                force[0] = -gradient.real() * body_ptr->mass;
                force[1] = -gradient.imag() * body_ptr->mass;
                
                forces[body_idx] += force;
            } else {
                // Improved 3D implementation with better numerical accuracy
                double potential = node->local.coeff[0].real();
                
                // Get direction and distance to source
                Vector<D> direction;
                double distance = 1.0;
                
                // Extract direction from local expansion with checks
                if (node->local.coeff.size() > 2 &&
                   (std::abs(node->local.coeff[1].real()) > 1e-14 || 
                    std::abs(node->local.coeff[1].imag()) > 1e-14 || 
                    std::abs(node->local.coeff[2].real()) > 1e-14)) {
                    
                    direction[0] = node->local.coeff[1].real();
                    direction[1] = node->local.coeff[1].imag();
                    direction[2] = node->local.coeff[2].real();
                    
                    // Get stored distance or calculate it
                    distance = node->local.coeff[2].imag();
                    if (distance < 1e-12) {
                        distance = direction.magnitude();
                        if (distance > 1e-12) {
                            direction = direction / distance; // Normalize
                        } else {
                            // Default direction if too close - use std::array constructor instead
                            direction = Vector<D>(std::array<double, D>{1.0, 0.0, 0.0});
                            distance = 1.0;
                        }
                    } else {
                        // Direction should already be normalized
                    }
                    
                    // Calculate gravitational force with quadrupole corrections
                    double force_magnitude = G * body_ptr->mass * potential / (distance * distance);
                    
                    // Apply quadrupole corrections when available
                    if (order > 2) {
                        double quad_correction = 0.0;
                        
                        for (int d1 = 0; d1 < D; ++d1) {
                            for (int d2 = d1; d2 < D; ++d2) {
                                int idx = 3 + d1 * D + d2;
                                if (idx < node->local.coeff.size()) {
                                    double quad_term = node->local.coeff[idx].real();
                                    if (std::abs(quad_term) > 1e-14) {
                                        // Direction-dependent quadrupole correction
                                        quad_correction += quad_term * direction[d1] * direction[d2];
                                    }
                                }
                            }
                        }
                        
                        // Apply correction to force magnitude
                        force_magnitude *= (1.0 + quad_correction);
                    }
                    
                    // Final force calculation
                    forces[body_idx] -= direction * force_magnitude;
                }
            }
        }
    }
}

template <int D>
void FMM_OMP<D>::p2p_phase(std::vector<Vector<D>>& forces, const std::vector<Body<D>>& bodies) {
    if (nodes_by_level.empty()) return;
    
    // Get leaf nodes
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    // Parallelize P2P computation for all leaf nodes with improved edge case handling
    #pragma omp parallel for
    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        FMMNode<D>* node = leaf_nodes[i];
        if (!node || node->bodies.empty()) continue;
        
        for (Body<D>* body_ptr : node->bodies) {
            size_t body_idx = 0;
            for (size_t j = 0; j < bodies.size(); ++j) {
                if (&bodies[j] == body_ptr) {
                    body_idx = j;
                    break;
                }
            }
            
            // Self-node interactions with improved close-range handling
            for (Body<D>* other : node->bodies) {
                if (other == body_ptr) continue;
                
                Vector<D> diff = other->position - body_ptr->position;
                double dist_sq = diff.magnitude_squared();
                
                // Improved handling of very close interactions
                if (dist_sq < 1e-10) {
                    // Apply Plummer softening for close interactions
                    const double epsilon = 1e-5;  // Softening parameter
                    dist_sq += epsilon * epsilon;
                }
                
                double dist = std::sqrt(dist_sq);
                // Use smoothed force calculation for better accuracy
                double force_mag = G * body_ptr->mass * other->mass / (dist_sq * dist);
                
                forces[body_idx] += diff.normalized() * force_mag;
            }
            
            // Neighbor-node interactions
            for (FMMNode<D>* neighbor : node->neighbor_list) {
                if (!neighbor) continue;
                
                for (Body<D>* other : neighbor->bodies) {
                    Vector<D> diff = other->position - body_ptr->position;
                    double dist_sq = diff.magnitude_squared();
                    
                    // Improved handling of very close interactions
                    if (dist_sq < 1e-10) {
                        // Apply Plummer softening for close interactions
                        const double epsilon = 1e-5;  // Softening parameter
                        dist_sq += epsilon * epsilon;
                    }
                    
                    double dist = std::sqrt(dist_sq);
                    // Use smoothed force calculation for better accuracy
                    double force_mag = G * body_ptr->mass * other->mass / (dist_sq * dist);
                    
                    forces[body_idx] += diff.normalized() * force_mag;
                }
            }
        }
    }
}

template <int D>
std::vector<Vector<D>> FMM_OMP<D>::calculate_forces(const std::vector<Body<D>>& bodies) {
    std::vector<Vector<D>> forces(bodies.size(), Vector<D>());
    
    if (!root) return forces;
    
    // Execute the optimized parallel FMM algorithm
    p2m_phase();   // Compute multipole expansions for leaf nodes
    m2m_phase();   // Translate multipole expansions up the tree
    m2l_phase();   // Translate multipole expansions to local expansions
    l2l_phase();   // Translate local expansions down the tree
    l2p_phase(forces, bodies);  // Evaluate local expansions
    p2p_phase(forces, bodies);  // Compute direct interactions
    
    return forces;
}

#endif // FMM_OMP_TPP
