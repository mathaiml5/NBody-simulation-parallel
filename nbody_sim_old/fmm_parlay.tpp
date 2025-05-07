// FMM_Parlay.cpp

#include "FMM_Parlay.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/sequence.h>

// P2M phase
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

    // Parallelize P2M computation using parlay::parallel_for
    parlay::parallel_for(0, parlay_leaf_nodes.size(), [&](size_t i) {
        FMMNode<D>* node = parlay_leaf_nodes[i];
        if (!node) return;

        // Clear multipole
        node->multipole.clear();

        double total_mass = 0.0;
        Vector<D> center_of_mass = Vector<D>();

        for (Body<D>* body : node->bodies) {
            total_mass += body->mass;
            center_of_mass += body->position * body->mass;
        }

        if (total_mass > 0) {
            center_of_mass = center_of_mass / total_mass;
        }

        // Store total mass
        node->multipole.coeff[0] = std::complex<double>(total_mass, 0.0);

        // Store center of mass
        if (order > 0) {
            std::complex<double> z_com = to_complex(center_of_mass - node->center);
            node->multipole.coeff[1] = z_com * total_mass;
        }

        // Compute higher-order terms
        for (Body<D>* body : node->bodies) {
            std::complex<double> z = to_complex(body->position - node->center);
            std::complex<double> z_power = z;

            for (int p = 2; p <= order; ++p) {
                node->multipole.coeff[p] += -body->mass * z_power / static_cast<double>(p);
                z_power *= z;
            }
        }
    });
}

// M2L phase
template <int D>
void FMM_Parlay<D>::m2l_phase() {
    if (nodes_by_level.empty()) return;

    // Process all nodes with interaction lists
    std::vector<FMMNode<D>*> nodes_with_interactions;

    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && !node->interaction_list.empty()) {
                nodes_with_interactions.push_back(node);
            }
        }
    }

    // Convert to parlay::sequence for parallel processing
    parlay::sequence<FMMNode<D>*> parlay_nodes(nodes_with_interactions.begin(), nodes_with_interactions.end());

    // Parallelize M2L computations
    parlay::parallel_for(0, parlay_nodes.size(), [&](size_t i) {
        FMMNode<D>* node = parlay_nodes[i];

        // Clear local expansion
        node->local.clear();

        for (FMMNode<D>* source : node->interaction_list) {
            if (!source) continue;

            Vector<D> r_vec = source->center - node->center;
            double r = r_vec.magnitude();

            if (r < 1e-10) continue;

            double src_mass = source->multipole.coeff[0].real();
            node->local.coeff[0] += std::complex<double>(src_mass / r, 0.0);
        }
    });
}

// L2P phase
template <int D>
void FMM_Parlay<D>::l2p_phase(parlay::sequence<Vector<D>>& forces, const parlay::sequence<Body<D>>& bodies) {
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

    for (FMMNode<D>* leaf : leaf_nodes) {
        for (Body<D>* body_ptr : leaf->bodies) {
            for (size_t i = 0; i < std_bodies.size(); ++i) {
                if (body_ptr->position == std_bodies[i].position) {
                    body_to_leaf[i] = leaf;
                    break;
                }
            }
        }
    }

    // Parallelize L2P computation
    parlay::parallel_for(0, bodies.size(), [&](size_t i) {
        FMMNode<D>* leaf = body_to_leaf[i];
        if (!leaf) return;

        const Body<D>& body = bodies[i];
        Vector<D> force;

        for (int p = 1; p <= order; ++p) {
            force += leaf->local.coeff[p].real() * body.mass;
        }

        forces[i] += force;
    });
}

// P2P phase
template <int D>
void FMM_Parlay<D>::p2p_phase(parlay::sequence<Vector<D>>& forces, const parlay::sequence<Body<D>>& bodies) {
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

    for (FMMNode<D>* leaf : leaf_nodes) {
        for (Body<D>* body_ptr : leaf->bodies) {
            for (size_t i = 0; i < std_bodies.size(); ++i) {
                if (body_ptr->position == std_bodies[i].position) {
                    body_to_leaf[i] = leaf;
                    break;
                }
            }
        }
    }

    // Parallelize P2P computation
    parlay::parallel_for(0, bodies.size(), [&](size_t i) {
        FMMNode<D>* leaf = body_to_leaf[i];
        if (!leaf) return;

        const Body<D>& body = bodies[i];

        for (Body<D>* other : leaf->bodies) {
            if (body.position == other->position) continue;

            Vector<D> diff = other->position - body.position;
            double dist_sq = diff.magnitude_squared();
            double dist = std::sqrt(dist_sq);
            double force_mag = G * body.mass * other->mass / (dist_sq * dist);

            forces[i] += diff.normalized() * force_mag;
        }
    });
}

// Calculate forces
template <int D>
parlay::sequence<Vector<D>> FMM_Parlay<D>::calculate_forces(const parlay::sequence<Body<D>>& bodies) {
    parlay::sequence<Vector<D>> forces(bodies.size());

    if (!root) return forces;

    p2m_phase();
    m2l_phase();
    l2p_phase(forces, bodies);
    p2p_phase(forces, bodies);

    return forces;
}

// Explicit template instantiations
template class FMM_Parlay<2>;
template class FMM_Parlay<3>;