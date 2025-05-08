#ifndef FMM_OMP_TPP
#define FMM_OMP_TPP

#include "fmm_omp.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <stdexcept>

template <int D>
FMM_OMP<D>::FMM_OMP(const std::vector<Body<D>>& bodies, 
                   int max_bodies, 
                   int max_lvl, 
                   int p)
    : max_bodies_per_leaf(max_bodies), max_level(max_lvl), order(p)
{
    FMM<D> fmm_builder(bodies, max_bodies, max_lvl, p);
    root = std::move(fmm_builder.root);
    build_level_lists();
    build_interaction_lists();
}

template <int D>
void FMM_OMP<D>::build_level_lists() {
    if (!root) return;
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
    nodes_by_level.resize(max_depth + 1);
    std::function<void(FMMNode<D>*, int)> collect_nodes;
    collect_nodes = [&](FMMNode<D>* node, int depth) {
        if (!node) return;
        nodes_by_level[depth].push_back(node);
        for (auto& child : node->children) {
            if (child) collect_nodes(child.get(), depth + 1);
        }
    };
    collect_nodes(root.get(), 0);
}

template <int D>
void FMM_OMP<D>::build_interaction_lists() {
    if (!root) return;
    std::function<void(FMMNode<D>*)> process;
    process = [&](FMMNode<D>* node) {
        if (!node) return;
        node->interaction_list.clear();
        node->neighbor_list.clear();
        if (node->is_leaf()) {
            std::function<void(FMMNode<D>*)> check_node;
            check_node = [&](FMMNode<D>* other) {
                if (!other || other == node) return;
                Vector<D> dist_vec = other->center - node->center;
                double dist = dist_vec.magnitude();
                double size_sum = node->half_size + other->half_size;
                if (dist > 2.5 * size_sum || node->half_size / dist < 0.3) {
                    node->interaction_list.push_back(other);
                } else if (other->is_leaf()) {
                    node->neighbor_list.push_back(other);
                } else {
                    for (auto& child : other->children) {
                        if (child) check_node(child.get());
                    }
                }
            };
            if (root && root.get() != node) {
                check_node(root.get());
            }
        }
        for (auto& child : node->children) {
            if (child) process(child.get());
        }
    };
    process(root.get());
}

template <int D>
void FMM_OMP<D>::p2m_phase() {
    if (nodes_by_level.empty()) return;
    std::vector<FMMNode<D>*> leaf_nodes;
    for (size_t level = 0; level < nodes_by_level.size(); ++level) {
        for (FMMNode<D>* node : nodes_by_level[level]) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        try {
            FMMNode<D>* node = leaf_nodes[i];
            if (!node) continue;
            for (int p = 0; p <= order; ++p) {
                node->multipole.coeff[p] = std::complex<double>(0.0, 0.0);
            }
            if constexpr (D == 2) {
                double total_mass = 0.0;
                Vector<D> center_of_mass = Vector<D>();
                for (Body<D>* body : node->bodies) {
                    total_mass += body->mass;
                    center_of_mass += body->position * body->mass;
                }
                if (total_mass > 1e-14) {
                    center_of_mass = center_of_mass / total_mass;
                } else {
                    center_of_mass = node->center;
                    total_mass = 1e-14;
                }
                node->multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
                if (order > 0) {
                    std::complex<double> z_com = to_complex(center_of_mass - node->center);
                    node->multipole.coeff[1] = z_com * total_mass;
                }
                for (Body<D>* body : node->bodies) {
                    std::complex<double> z = to_complex(body->position - center_of_mass);
                    std::complex<double> z_power = z;
                    for (int p = 2; p <= order; ++p) {
                        z_power *= z;
                        double norm_factor = 1.0 / static_cast<double>(::factorial(p));
                        node->multipole.coeff[p] += body->mass * z_power * norm_factor;
                    }
                }
            } else {
                double total_mass = 0.0;
                Vector<D> center_of_mass = Vector<D>();
                for (Body<D>* body : node->bodies) {
                    total_mass += body->mass;
                    center_of_mass += body->position * body->mass;
                }
                if (total_mass > 1e-14) {
                    center_of_mass = center_of_mass / total_mass;
                } else {
                    center_of_mass = node->center;
                    total_mass = 1e-14;
                }
                node->multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
                if (order > 0 && node->multipole.coeff.size() > 2) {
                    node->multipole.coeff[1] = std::complex<double>(center_of_mass[0], center_of_mass[1]);
                    node->multipole.coeff[2] = std::complex<double>(center_of_mass[2], 0.0);
                }
                for (Body<D>* body : node->bodies) {
                    Vector<D> r_vec = body->position - center_of_mass;
                    double r_squared = r_vec.magnitude_squared();
                    if (r_squared > 1e-10 && order > 2) {
                        for (int d1 = 0; d1 < D; ++d1) {
                            for (int d2 = d1; d2 < D; ++d2) {
                                int idx = 3 + d1 * D + d2;
                                if (idx < node->multipole.coeff.size()) {
                                    double quad_term = 0.5 * body->mass * 
                                        (3.0 * r_vec[d1] * r_vec[d2] - (d1 == d2 ? r_squared : 0.0));
                                    node->multipole.coeff[idx] += std::complex<double>(quad_term, 0.0);
                                }
                            }
                        }
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
        catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Exception in P2M phase: " << e.what() << std::endl;
            }
        }
        catch (...) {
            #pragma omp critical
            {
                std::cerr << "Unknown error in P2M phase" << std::endl;
            }
        }
    }
}

template <int D>
void FMM_OMP<D>::m2m_phase() {
    if (nodes_by_level.empty()) return;
    try {
        for (int level = static_cast<int>(nodes_by_level.size()) - 1; level > 0; --level) {
            std::vector<FMMNode<D>*>& nodes = nodes_by_level[level];
            std::vector<FMMNode<D>*> parent_nodes;
            std::vector<bool> parent_processed(nodes_by_level[level-1].size(), false);
            for (FMMNode<D>* node : nodes) {
                if (!node || !node->parent) continue;
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
            for (FMMNode<D>* parent : parent_nodes) {
                if (!parent) continue;
                for (int p = 0; p <= order; ++p) {
                    parent->multipole.coeff[p] = std::complex<double>(0.0, 0.0);
                }
            }
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < parent_nodes.size(); ++i) {
                try {
                    FMMNode<D>* node = parent_nodes[i];
                    if (!node) continue;
                    MultipoleExpansion thread_local_multipole;
                    for (int p = 0; p <= order; ++p) {
                        thread_local_multipole.coeff[p] = std::complex<double>(0.0, 0.0);
                    }
                    for (auto& child : node->children) {
                        if (!child) continue;
                        if constexpr (D == 2) {
                            std::complex<double> z0 = to_complex(child->center - node->center);
                            thread_local_multipole.coeff[0] += child->multipole.coeff[0];
                            for (int p = 1; p <= order; ++p) {
                                thread_local_multipole.coeff[p] += child->multipole.coeff[p];
                                for (int k = 0; k < p; ++k) {
                                    if (std::abs(child->multipole.coeff[k]) < 1e-14) continue;
                                    std::complex<double> term = 
                                        child->multipole.coeff[k] * pow(-z0, p-k) * 
                                        static_cast<double>(binomial(p, k));
                                    if (std::isfinite(term.real()) && std::isfinite(term.imag())) {
                                        thread_local_multipole.coeff[p] += term;
                                    }
                                }
                            }
                        }
                        else {
                            thread_local_multipole.coeff[0] += child->multipole.coeff[0];
                            if (thread_local_multipole.coeff.size() > 2 && 
                                child->multipole.coeff.size() > 2) {
                                thread_local_multipole.coeff[1] = child->multipole.coeff[1];
                                thread_local_multipole.coeff[2] = child->multipole.coeff[2];
                            }
                        }
                    }
                    #pragma omp critical
                    {
                        for (int p = 0; p <= order; ++p) {
                            node->multipole.coeff[p] += thread_local_multipole.coeff[p];
                        }
                    }
                }
                catch (const std::exception& e) {
                    #pragma omp critical
                    {
                        std::cerr << "Exception in M2M phase: " << e.what() << std::endl;
                    }
                }
                catch (...) {
                    #pragma omp critical
                    {
                        std::cerr << "Unknown error in M2M phase" << std::endl;
                    }
                }
            }
        }
    }
    catch (...) {
        std::cerr << "Error in M2M phase" << std::endl;
    }
}

template <int D>
std::vector<Vector<D>> FMM_OMP<D>::calculate_forces(const std::vector<Body<D>>& bodies) {
    std::vector<Vector<D>> forces(bodies.size(), Vector<D>());
    if (!root) return forces;
    try {
        p2m_phase();
        m2m_phase();
        m2l_phase();
        l2l_phase();
        l2p_phase(forces, bodies);
        p2p_phase(forces, bodies);
        bool has_invalid = false;
        for (const auto& force : forces) {
            for (int d = 0; d < D; ++d) {
                if (!std::isfinite(force[d])) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) break;
        }
        if (has_invalid) {
            std::cerr << "Warning: Invalid values detected in FMM-OpenMP. Using fallback." << std::endl;
            std::fill(forces.begin(), forces.end(), Vector<D>());
            FMM<D> fallback(bodies, max_bodies_per_leaf, max_level, order);
            return fallback.calculate_forces(bodies);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in FMM-OpenMP: " << e.what() << ". Using fallback." << std::endl;
        std::fill(forces.begin(), forces.end(), Vector<D>());
        FMM<D> fallback(bodies, max_bodies_per_leaf, max_level, order);
        return fallback.calculate_forces(bodies);
    }
    catch (...) {
        std::cerr << "Unknown error in FMM-OpenMP. Using fallback." << std::endl;
        std::fill(forces.begin(), forces.end(), Vector<D>());
        FMM<D> fallback(bodies, max_bodies_per_leaf, max_level, order);
        return fallback.calculate_forces(bodies);
    }
    return forces;
}

#endif // FMM_OMP_TPP
