#ifndef FMM_PARLAY_TPP
#define FMM_PARLAY_TPP

#include "fmm_parlay.h"
#include <algorithm>
#include <functional>

template <int D>
FMM_Parlay<D>::FMM_Parlay(const parlay::sequence<Body<D>>& bodies, 
                   int max_bodies, 
                   int max_lvl, 
                   int p)
    : max_bodies_per_leaf(max_bodies), max_level(max_lvl), order(p)
{
    // Convert parlay::sequence to std::vector for tree building
    std::vector<Body<D>> std_bodies(bodies.begin(), bodies.end());
    
    // Create a regular FMM object to build the tree and interaction lists
    FMM<D> fmm_builder(std_bodies, max_bodies, max_lvl, p);
    
    // Extract the root
    root = std::move(fmm_builder.root);
    
    // Build level lists for parallel processing
    build_level_lists();
    
    // Rebuild interaction lists with more conservative criteria
    build_interaction_lists();
}

template <int D>
void FMM_Parlay<D>::build_level_lists() {
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

// P2M phase optimized with ParlayLib
template <int D>
void FMM_Parlay<D>::p2m_phase() {
    if (nodes_by_level.empty()) return;
    
    // Get leaf nodes
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (size_t level = 0; level < nodes_by_level.size(); ++level) {
        for (FMMNode<D>* node : nodes_by_level[level]) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    // Convert to parlay::sequence for parallel processing
    parlay::sequence<FMMNode<D>*> parlay_leaf_nodes(leaf_nodes.begin(), leaf_nodes.end());
    
    // Parallelize P2M computation using parlay::parallel_for with improved accuracy
    parlay::parallel_for(0, parlay_leaf_nodes.size(), [&](size_t i) {
        FMMNode<D>* node = parlay_leaf_nodes[i];
        if (!node) return;
        
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
            // Enhanced 3D implementation
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
    });
}

// M2M phase optimized with ParlayLib (hierarchical parallelism)
template <int D>
void FMM_Parlay<D>::m2m_phase() {
    if (!root) return;
    
    // Process levels from bottom up, skip leaf level
    for (int level = static_cast<int>(nodes_by_level.size()) - 1; level > 0; --level) {
        auto& level_nodes = nodes_by_level[level];
        
        // Get parent nodes that need M2M computation
        std::vector<FMMNode<D>*> parent_nodes;
        std::vector<bool> parent_processed(nodes_by_level[level-1].size(), false);
        
        for (FMMNode<D>* node : level_nodes) {
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
        
        // Convert to parlay::sequence for parallel processing
        parlay::sequence<FMMNode<D>*> parlay_parent_nodes(parent_nodes.begin(), parent_nodes.end());
        
        // Parallelize M2M computation using parlay::parallel_for
        parlay::parallel_for(0, parlay_parent_nodes.size(), [&](size_t i) {
            FMMNode<D>* node = parlay_parent_nodes[i];
            
            // Clear multipole first
            node->multipole.clear();
            
            // Accumulate from all children with improved numerical stability
            for (auto& child : node->children) {
                if (child) {
                    // M2M: Child multipole to parent multipole
                    if constexpr (D == 2) {
                        // More accurate calculation of the translation vector
                        Vector<D> child_com = child->center;
                        if (std::abs(child->multipole.coeff[1]) > 1e-12) {
                            std::complex<double> z_com = child->multipole.coeff[1] / child->multipole.coeff[0].real();
                            child_com[0] += z_com.real();
                            child_com[1] += z_com.imag();
                        }
                        
                        // Use vector from parent center to child's center of mass
                        Vector<D> translation_vec = child_com - node->center;
                        std::complex<double> z0 = to_complex(translation_vec);
                        
                        // Add mass term
                        node->multipole.coeff[0] += child->multipole.coeff[0];
                        
                        // Enhanced higher order terms with better numerical stability
                        for (int p = 1; p <= order; ++p) {
                            // Direct transfer of child's term
                            if (std::abs(child->multipole.coeff[p]) > 1e-12) {
                                node->multipole.coeff[p] += child->multipole.coeff[p];
                            }
                            
                            // Translation terms
                            for (int k = 0; k < p; ++k) {
                                if (std::abs(child->multipole.coeff[k]) < 1e-12) continue;
                                
                                double bin_coeff = binomial(p, k);
                                std::complex<double> term = child->multipole.coeff[k] * 
                                    std::pow(-z0, p-k) * bin_coeff;
                                
                                if (std::abs(term) > 1e-14) {
                                    node->multipole.coeff[p] += term;
                                }
                            }
                        }
                    } else {
                        // Enhanced 3D implementation
                        // Add mass term
                        node->multipole.coeff[0] += child->multipole.coeff[0];
                        
                        // Add COM information with improved handling
                        if (node->multipole.coeff.size() > 2 && child->multipole.coeff.size() > 2) {
                            double child_mass = child->multipole.coeff[0].real();
                            if (child_mass > 1e-12) {
                                double current_mass = node->multipole.coeff[0].real() - child_mass;
                                
                                // Extract child's center of mass
                                Vector<D> child_com;
                                child_com[0] = child->multipole.coeff[1].real();
                                child_com[1] = child->multipole.coeff[1].imag();
                                child_com[2] = child->multipole.coeff[2].real();
                                
                                // Calculate translation vector with better accuracy
                                Vector<D> r_vec = child_com - node->center;
                                
                                if (current_mass > 1e-12) {
                                    // Update COM weighted by masses
                                    Vector<D> current_com;
                                    current_com[0] = node->multipole.coeff[1].real();
                                    current_com[1] = node->multipole.coeff[1].imag();
                                    current_com[2] = node->multipole.coeff[2].real();
                                    
                                    double new_mass = current_mass + child_mass;
                                    Vector<D> new_com;
                                    
                                    for (int d = 0; d < D; ++d) {
                                        new_com[d] = (current_com[d] * current_mass + child_com[d] * child_mass) / new_mass;
                                    }
                                    
                                    node->multipole.coeff[1] = std::complex<double>(new_com[0], new_com[1]);
                                    node->multipole.coeff[2] = std::complex<double>(new_com[2], 0.0);
                                } else {
                                    // If this was the first child, just copy its COM
                                    node->multipole.coeff[1] = child->multipole.coeff[1];
                                    node->multipole.coeff[2] = child->multipole.coeff[2];
                                }
                                
                                // Translate higher-order moments with improved stability
                                if (order > 2) {
                                    for (int idx = 3; idx < std::min(child->multipole.coeff.size(), node->multipole.coeff.size()); ++idx) {
                                        // Only add significant terms
                                        if (std::abs(child->multipole.coeff[idx]) > 1e-14) {
                                            node->multipole.coeff[idx] += child->multipole.coeff[idx];
                                        }
                                        
                                        // Apply translation corrections for quadrupole moments
                                        if (idx >= 3 && idx < 12 && child_mass > 1e-12) {
                                            // Extract quadrupole indices
                                            int d1 = (idx - 3) / D;
                                            int d2 = (idx - 3) % D;
                                            
                                            // Translation correction
                                            double correction = child_mass * r_vec[d1] * r_vec[d2];
                                            node->multipole.coeff[idx] += std::complex<double>(correction, 0.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

template <int D>
void FMM_Parlay<D>::build_interaction_lists() {
    if (!root) return;
    
    // Use more conservative multipole acceptance criterion (0.3 instead of 0.5)
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
                
                // More conservative criterion for well-separateness
                double size_sum = node->half_size + other->half_size;
                
                // Use 0.3 instead of 0.5, or a more conservative distance threshold
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
    
    // Use parlay::par_do for the top level to parallelize list building
    parlay::par_do(
        [&]() { process(root.get()); },
        []() { /* No-op second task */ }
    );
}

// M2L phase optimized with ParlayLib
template <int D>
void FMM_Parlay<D>::m2l_phase() {
    if (nodes_by_level.empty()) return;
    
    // Loop over levels and process nodes in parallel
    for (auto& level_nodes : nodes_by_level) {
        // First find all nodes with interactions
        auto nodes_with_interactions = parlay::filter(level_nodes, [](FMMNode<D>* node) {
            return node && !node->interaction_list.empty();
        });
        
        if (nodes_with_interactions.empty()) continue;
        
        // Process nodes in parallel
        parlay::parallel_for(0, nodes_with_interactions.size(), [&](size_t i) {
            FMMNode<D>* node = nodes_with_interactions[i];
            if (!node) return;
            
            // Convert to parlay sequence for better parallelization
            parlay::sequence<FMMNode<D>*> interaction_list(
                node->interaction_list.begin(), 
                node->interaction_list.end()
            );
            
            // Choose appropriate strategy based on list size
            if (adaptive_parallelism) {
                m2l_adaptive(node, interaction_list);
            } else if (interaction_list.size() > 32) {
                m2l_parallel(node, interaction_list);
            } else {
                m2l_sequential(node, interaction_list);
            }
        });
    }
}

template <int D>
void FMM_Parlay<D>::m2l_adaptive(FMMNode<D>* node, parlay::sequence<FMMNode<D>*>& interaction_list) {
    // Adaptively choose strategy based on interaction list size
    if (interaction_list.size() > 64) {
        m2l_parallel(node, interaction_list);
    } else {
        m2l_sequential(node, interaction_list);
    }
}

template <int D>
void FMM_Parlay<D>::m2l_sequential(FMMNode<D>* node, parlay::sequence<FMMNode<D>*>& interaction_list) {
    // Clear local expansion first
    node->local.clear();
    
    // Process all sources sequentially - better for small lists
    for (FMMNode<D>* source : interaction_list) {
        double src_mass = source->multipole.coeff[0].real();
        if (std::abs(src_mass) < EPSILON) continue;  // Skip if mass too small
        
        if constexpr (D == 2) {
            // 2D implementation for M2L
            // Extract accurate center of mass
            Vector<D> source_com = source->center;
            if (order > 0 && std::abs(source->multipole.coeff[1]) > 1e-12) {
                std::complex<double> z_com = source->multipole.coeff[1] / src_mass;
                source_com[0] += z_com.real();
                source_com[1] += z_com.imag();
            }
            
            // Improved vector calculation using COM
            Vector<D> r_vec = source_com - node->center;
            double r = r_vec.magnitude();
            
            if (r < EPSILON) continue;
            
            // Use the improved r_vec based on center of mass
            std::complex<double> z0 = to_complex(r_vec);
            
            // Enhanced and more stable M2L translation
            for (int p = 0; p <= order; ++p) {
                // Use more accurate monopole approximation (p=0)
                if (p == 0) {
                    node->local.coeff[0] += std::complex<double>(src_mass / r, 0.0);
                    continue;
                }
                
                for (int q = 0; q <= order-p; ++q) {
                    std::complex<double> term;
                    if (q == 0) {
                        // Improved monopole term
                        term = src_mass * std::pow(1.0 / std::max(r, EPSILON), p);
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
            // 3D implementation - with improved accuracy
            Vector<D> source_com = source->center;
            if (source->multipole.coeff.size() > 2) {
                source_com[0] = source->multipole.coeff[1].real();
                source_com[1] = source->multipole.coeff[1].imag();
                source_com[2] = source->multipole.coeff[2].real();
            }
            
            Vector<D> r_com_vec = source_com - node->center;
            double r_com = r_com_vec.magnitude();
            
            if (r_com < EPSILON) continue;
            
            Vector<D> r_norm = r_com_vec / r_com;
            
            node->local.coeff[0] += std::complex<double>(src_mass / r_com, 0.0);
            
            if (node->local.coeff.size() > 2) {
                node->local.coeff[1] = std::complex<double>(r_norm[0], r_norm[1]);
                node->local.coeff[2] = std::complex<double>(r_norm[2], r_com);
            }
            
            if (use_quadrupole_correction && order > 2) {
                double r_com_sq = r_com * r_com;
                double r_com_5 = r_com_sq * r_com_sq * r_com;
                
                for (int d1 = 0; d1 < D; ++d1) {
                    for (int d2 = d1; d2 < D; ++d2) {
                        int idx = 3 + d1 * D + d2;
                        if (idx < source->multipole.coeff.size()) {
                            double quad_term = source->multipole.coeff[idx].real();
                            
                            if (std::abs(quad_term) > EPSILON) {
                                double contrib = 0.5 * quad_term / r_com_5;
                                
                                double dir_factor = 5.0 * r_norm[d1] * r_norm[d2];
                                if (d1 == d2) dir_factor -= 1.0;
                                contrib *= dir_factor;
                                
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

template <int D>
void FMM_Parlay<D>::m2l_parallel(FMMNode<D>* node, parlay::sequence<FMMNode<D>*>& interaction_list) {
    // Create local expansion for the node
    node->local.clear();
    
    // For large lists, process in parallel for better performance
    size_t n = interaction_list.size();
    
    // Create temporary local expansions, one for each batch
    size_t num_batches = std::min(size_t(32), n);
    
    // Create a vector of temporary local expansions - fix instantiation to match the one in fmm.h
    std::vector<Expansion<D, 32>> temp_expansions(num_batches); // Use 32 for max order (P parameter)
    
    // Initialize local expansions - fix to use the constructor directly instead of resize
    for (size_t i = 0; i < num_batches; i++) {
        temp_expansions[i] = Expansion<D, 32>();
    }
    
    // Process interactions in batches to reduce contention
    parlay::parallel_for(0, num_batches, [&](size_t batch) {
        // Compute start and end indices for this batch
        size_t start = (batch * n) / num_batches;
        size_t end = ((batch + 1) * n) / num_batches;
        
        // Process assigned sources
        for (size_t j = start; j < end; j++) {
            FMMNode<D>* source = interaction_list[j];
            if (!source) continue;
            
            double src_mass = source->multipole.coeff[0].real();
            if (std::abs(src_mass) < EPSILON) continue;
            
            if constexpr (D == 2) {
                // 2D implementation for M2L in parallel
                // Extract accurate center of mass
                Vector<D> source_com = source->center;
                if (order > 0 && std::abs(source->multipole.coeff[1]) > 1e-12) {
                    std::complex<double> z_com = source->multipole.coeff[1] / src_mass;
                    source_com[0] += z_com.real();
                    source_com[1] += z_com.imag();
                }
                
                // Improved vector calculation using COM
                Vector<D> r_vec = source_com - node->center;
                double r = r_vec.magnitude();
                
                if (r < EPSILON) continue;
                
                // Use the improved r_vec based on center of mass
                std::complex<double> z0 = to_complex(r_vec);
                
                // Add monopole term (p=0)
                temp_expansions[batch].coeff[0] += std::complex<double>(src_mass / r, 0.0);
                
                // Enhanced and more stable M2L translation for higher order terms
                for (int p = 1; p <= order; ++p) {
                    for (int q = 0; q <= order-p; ++q) {
                        std::complex<double> term;
                        if (q == 0) {
                            // Improved monopole term
                            term = src_mass * std::pow(1.0 / std::max(r, EPSILON), p);
                        } else {
                            // Higher order with better numerical stability
                            double bin_coeff = binomial(p+q-1, q-1);
                            // Improved accuracy with direct division
                            std::complex<double> factor = source->multipole.coeff[q] / std::pow(z0, p+q);
                            term = bin_coeff * factor;
                        }
                        
                        // Add term if significant
                        if (std::abs(term) > 1e-14 && p < temp_expansions[batch].coeff.size()) {
                            temp_expansions[batch].coeff[p] += term;
                        }
                    }
                }
            } else {
                // 3D implementation with better accuracy and error handling
                // Extract center of mass if available
                Vector<D> source_com = source->center;
                if (source->multipole.coeff.size() > 2) {
                    source_com[0] = source->multipole.coeff[1].real();
                    source_com[1] = source->multipole.coeff[1].imag();
                    source_com[2] = source->multipole.coeff[2].real();
                }
                
                // Vector from target to source COM
                Vector<D> r_com_vec = source_com - node->center;
                double r_com = r_com_vec.magnitude();
                
                // Skip if too close
                if (r_com < EPSILON) continue;
                
                // Calculate normalized direction vector 
                Vector<D> r_norm = r_com_vec / r_com;
                
                // Add monopole contribution to this batch's local expansion
                temp_expansions[batch].coeff[0] += std::complex<double>(src_mass / r_com, 0.0);
                
                // Store direction information (batch-specific)
                if (temp_expansions[batch].coeff.size() > 2) {
                    temp_expansions[batch].coeff[1] = std::complex<double>(r_norm[0], r_norm[1]);
                    temp_expansions[batch].coeff[2] = std::complex<double>(r_norm[2], r_com);
                }
                
                // Add quadrupole contribution if enabled
                if (use_quadrupole_correction && order > 2) {
                    double r_com_sq = r_com * r_com;
                    double r_com_5 = r_com_sq * r_com_sq * r_com;
                    
                    for (int d1 = 0; d1 < D; ++d1) {
                        for (int d2 = d1; d2 < D; ++d2) {
                            int idx = 3 + d1 * D + d2;
                            if (idx < source->multipole.coeff.size()) {
                                double quad_term = source->multipole.coeff[idx].real();
                                
                                if (std::abs(quad_term) > EPSILON) {
                                    double contrib = 0.5 * quad_term / r_com_5;
                                    double dir_factor = 5.0 * r_norm[d1] * r_norm[d2];
                                    if (d1 == d2) dir_factor -= 1.0;
                                    contrib *= dir_factor;
                                    
                                    if (idx < temp_expansions[batch].coeff.size()) {
                                        temp_expansions[batch].coeff[idx] += std::complex<double>(contrib, 0.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    
    // Combine results from all batches
    for (const auto& local : temp_expansions) {
        for (size_t i = 0; i < std::min(local.coeff.size(), node->local.coeff.size()); ++i) {
            node->local.coeff[i] += local.coeff[i];
        }
    }
}

template <int D>
void FMM_Parlay<D>::l2l_phase() {
    if (nodes_by_level.empty()) return;
    
    // Process levels from top down using parlay::par_do for hierarchical parallelism
    std::function<void(FMMNode<D>*)> l2l_recursive;
    l2l_recursive = [&](FMMNode<D>* node) {
        if (!node || node->is_leaf()) return;
        
        std::vector<FMMNode<D>*> valid_children;
        for (auto& child : node->children) {
            if (child) {
                valid_children.push_back(child.get());
                
                if constexpr (D == 2) {
                    std::complex<double> z0 = to_complex(node->center - child->center);
                    
                    for (int p = 0; p <= order; ++p) {
                        child->local.coeff[p] += node->local.coeff[p];
                        
                        for (int q = p + 1; q <= order; ++q) {
                            std::complex<double> term = node->local.coeff[q] * 
                                                     pow(z0, q - p) *
                                                     static_cast<double>(binomial(q, p));
                            child->local.coeff[p] += term;
                        }
                    }
                } else {
                    child->local.coeff[0] += node->local.coeff[0] / static_cast<double>((1 << D));
                    
                    if (child->local.coeff.size() > 2 && node->local.coeff.size() > 2) {
                        child->local.coeff[1] = node->local.coeff[1];
                        child->local.coeff[2] = node->local.coeff[2];
                    }
                }
            }
        }
        
        if (valid_children.empty()) return;
        
        if (valid_children.size() > 1) {
            parlay::par_do(
                [&]() { 
                    size_t mid = valid_children.size() / 2;
                    for (size_t i = 0; i < mid; i++) {
                        l2l_recursive(valid_children[i]);
                    }
                },
                [&]() { 
                    size_t mid = valid_children.size() / 2;
                    for (size_t i = mid; i < valid_children.size(); i++) {
                        l2l_recursive(valid_children[i]);
                    }
                }
            );
        } else {
            l2l_recursive(valid_children[0]);
        }
    };
    
    if (root) {
        l2l_recursive(root.get());
    }
}

template <int D>
void FMM_Parlay<D>::l2p_phase(parlay::sequence<Vector<D>>& forces, 
                            const parlay::sequence<Body<D>>& bodies) {
    if (nodes_by_level.empty() || bodies.empty()) return;
    
    std::vector<Body<D>> std_bodies(bodies.begin(), bodies.end());
    
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    std::vector<FMMNode<D>*> body_to_leaf(bodies.size(), nullptr);
    
    for (FMMNode<D>* leaf : leaf_nodes) {
        for (Body<D>* body_ptr : leaf->bodies) {
            for (size_t i = 0; i < std_bodies.size(); ++i) {
                bool is_match = true;
                for (int d = 0; d < D; ++d) {
                    if (std::abs(body_ptr->position[d] - std_bodies[i].position[d]) > 1e-9) {
                        is_match = false;
                        break;
                    }
                }
                if (is_match) {
                    body_to_leaf[i] = leaf;
                    break;
                }
            }
        }
    }
    
    // Parallelize L2P computation using parlay::parallel_for
    parlay::parallel_for(0, bodies.size(), [&](size_t i) {
        FMMNode<D>* leaf = body_to_leaf[i];
        if (!leaf) return;
        
        const Body<D>& body = bodies[i];
        
        if constexpr (D == 2) {
            // Improved gradient calculation for 2D
            std::complex<double> z = to_complex(body.position - leaf->center);
            std::complex<double> potential(0.0, 0.0);
            std::complex<double> gradient(0.0, 0.0);
            
            // Calculate potential and its gradient more accurately
            for (int p = 0; p <= order; ++p) {
                // Skip small coefficients for numerical stability
                if (std::abs(leaf->local.coeff[p]) < 1e-14) continue;
                
                std::complex<double> z_power = p > 0 ? std::pow(z, p-1) : std::complex<double>(1.0, 0.0);
                
                // Add to potential
                potential += leaf->local.coeff[p] * std::pow(z, p);
                
                // Add to gradient (derivative of potential)
                if (p > 0) {
                    gradient += static_cast<double>(p) * leaf->local.coeff[p] * z_power;
                }
            }
            
            // Force is negative gradient of potential
            Vector<D> force;
            force[0] = -gradient.real() * body.mass;
            force[1] = -gradient.imag() * body.mass;
            
            forces[i] += force;
        } else {
            // Improved 3D implementation with better numerical accuracy
            const Body<D>& body = bodies[i];
            double potential = leaf->local.coeff[0].real();
            
            // Get direction and distance to source
            Vector<D> direction;
            double distance = 1.0;
            
            // Extract direction from local expansion with checks
            if (leaf->local.coeff.size() > 2 && 
                (std::abs(leaf->local.coeff[1].real()) > 1e-14 || 
                 std::abs(leaf->local.coeff[1].imag()) > 1e-14 || 
                 std::abs(leaf->local.coeff[2].real()) > 1e-14)) {
                
                direction[0] = leaf->local.coeff[1].real();
                direction[1] = leaf->local.coeff[1].imag();
                direction[2] = leaf->local.coeff[2].real();
                
                // Get stored distance or calculate it
                distance = leaf->local.coeff[2].imag();
                if (distance < 1e-12) {
                    distance = direction.magnitude();
                    if (distance > 1e-12) {
                        direction = direction / distance;
                    } else {
                        std::array<double, D> default_dir;
                        default_dir[0] = 1.0;
                        for (int d = 1; d < D; d++) {
                            default_dir[d] = 0.0;
                        }
                        direction = Vector<D>(default_dir);
                        distance = 1.0;
                    }
                }
                
                double force_magnitude = G * body.mass * potential / (distance * distance);
                
                if (order > 2) {
                    double quad_correction = 0.0;
                    
                    for (int d1 = 0; d1 < D; ++d1) {
                        for (int d2 = d1; d2 < D; ++d2) {
                            int idx = 3 + d1 * D + d2;
                            if (idx < leaf->local.coeff.size()) {
                                double quad_term = leaf->local.coeff[idx].real();
                                if (std::abs(quad_term) > 1e-14) {
                                    // Direction-dependent quadrupole correction
                                    quad_correction += quad_term * direction[d1] * direction[d2];
                                }
                            }
                        }
                    }
                    
                    force_magnitude *= (1.0 + quad_correction);
                }
                
                // Final force calculation
                forces[i] -= direction * force_magnitude;
            }
        }
    });
}

// P2P phase optimized with ParlayLib
template <int D>
void FMM_Parlay<D>::p2p_phase(parlay::sequence<Vector<D>>& forces, 
                            const parlay::sequence<Body<D>>& bodies) {
    if (nodes_by_level.empty() || bodies.empty()) return;
    
    // Convert bodies to std::vector for tree traversal
    std::vector<Body<D>> std_bodies(bodies.begin(), bodies.end());
    
    // Get leaf nodes
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    // Map bodies to their leaf nodes for direct lookup
    std::vector<FMMNode<D>*> body_to_leaf(bodies.size(), nullptr);
    std::vector<size_t> body_indices(bodies.size(), 0);
    
    for (FMMNode<D>* leaf : leaf_nodes) {
        for (Body<D>* body_ptr : leaf->bodies) {
            // Find the index of this body in the original array
            for (size_t i = 0; i < std_bodies.size(); ++i) {
                // Compare positions to find matching body
                bool is_match = true;
                for (int d = 0; d < D; ++d) {
                    if (std::abs(body_ptr->position[d] - std_bodies[i].position[d]) > 1e-9) {
                        is_match = false;
                        break;
                    }
                }
                if (is_match) {
                    body_to_leaf[i] = leaf;
                    body_indices[i] = i;
                    break;
                }
            }
        }
    }
    
    struct P2PWork {
        size_t body_idx;
        FMMNode<D>* leaf;
        FMMNode<D>* neighbor;
    };
    
    std::vector<P2PWork> work_items;
    
    for (size_t i = 0; i < bodies.size(); ++i) {
        FMMNode<D>* leaf = body_to_leaf[i];
        if (!leaf) continue;
        
        // Add self interactions
        work_items.push_back({i, leaf, leaf});
        
        // Add neighbor interactions
        for (FMMNode<D>* neighbor : leaf->neighbor_list) {
            if (neighbor) {
                work_items.push_back({i, leaf, neighbor});
            }
        }
    }
    
    // Convert to parlay::sequence for parallel processing
    parlay::sequence<P2PWork> parlay_work(work_items.begin(), work_items.end());
    
    // Parallelize P2P computation with improved handling of edge cases
    parlay::parallel_for(0, parlay_work.size(), [&](size_t i) {
        P2PWork& work = parlay_work[i];
        size_t body_idx = work.body_idx;
        const Body<D>& body = bodies[body_idx];
        FMMNode<D>* neighbor = work.neighbor;
        
        for (const Body<D>* other : neighbor->bodies) {
            // Skip self interaction
            bool is_same = true;
            for (int d = 0; d < D; ++d) {
                if (std::abs(other->position[d] - body.position[d]) > 1e-14) {
                    is_same = false;
                    break;
                }
            }
            if (is_same) continue;
            
            // Improved direct force calculation
            Vector<D> diff = other->position - body.position;
            double dist_sq = diff.magnitude_squared();
            
            // Improved handling of very close interactions
            if (dist_sq < 1e-10) {
                const double epsilon = 1e-5;
                dist_sq += epsilon * epsilon;
            }
            
            double dist = std::sqrt(dist_sq);
            // Use smoothed force calculation for better accuracy
            double force_mag = G * body.mass * other->mass / (dist_sq * dist);
            
            forces[body_idx] += diff.normalized() * force_mag;
        }
    });
}

template <int D>
parlay::sequence<Vector<D>> FMM_Parlay<D>::calculate_forces(const parlay::sequence<Body<D>>& bodies) {
    parlay::sequence<Vector<D>> forces(bodies.size());
    
    if (!root || nodes_by_level.empty()) {
        std::cerr << "FMM error: Tree not properly constructed!" << std::endl;
        return forces;
    }
    
    try {
        p2m_phase();
        m2m_phase();
        m2l_phase();
        l2l_phase();
        l2p_phase(forces, bodies);
        p2p_phase(forces, bodies);
    } catch (const std::exception& e) {
        std::cerr << "FMM error during force calculation: " << e.what() << std::endl;
    }
    
    return forces;
}

#endif // FMM_PARLAY_TPP
