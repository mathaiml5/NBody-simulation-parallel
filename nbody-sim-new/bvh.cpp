#ifndef BVH_TPP
#define BVH_TPP

#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <unordered_map>
#include <limits>

#include "bvh.h"

// External gravitational constant
extern const double grav;

// BVH constructor implementation
template <int D>
BVH<D>::BVH(const std::vector<Body<D>>& bodies, int max_bodies)
    : max_bodies_per_leaf(max_bodies) { // Removed theta initialization
    if (bodies.empty()) return;
    
    // Convert to pointers for easier manipulation
    std::vector<Body<D>*> body_ptrs;
    body_ptrs.reserve(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
        body_ptrs.push_back(const_cast<Body<D>*>(&bodies[i]));
    }
    
    // Build tree recursively
    root = build_tree(body_ptrs, 0);
}

// Split bodies along the longest axis
template <int D>
std::pair<std::vector<Body<D>*>, std::vector<Body<D>*>> BVH<D>::split_bodies(
    std::vector<Body<D>*>& bodies) {
    
    // Find the bounds of all bodies
    Vector<D> min_bound = bodies[0]->position;
    Vector<D> max_bound = bodies[0]->position;
    
    for (size_t i = 1; i < bodies.size(); ++i) {
        for (int d = 0; d < D; ++d) {
            min_bound[d] = std::min(min_bound[d], bodies[i]->position[d]);
            max_bound[d] = std::max(max_bound[d], bodies[i]->position[d]);
        }
    }
    
    // Find the longest axis
    int longest_axis = 0;
    double max_length = 0;
    
    for (int d = 0; d < D; ++d) {
        double length = max_bound[d] - min_bound[d];
        if (length > max_length) {
            max_length = length;
            longest_axis = d;
        }
    }
    
    // Sort bodies along the longest axis
    std::sort(bodies.begin(), bodies.end(),
              [longest_axis](const Body<D>* a, const Body<D>* b) {
                  return a->position[longest_axis] < b->position[longest_axis];
              });
    
    // Split in the middle
    size_t mid = bodies.size() / 2;
    std::vector<Body<D>*> left_bodies(bodies.begin(), bodies.begin() + mid);
    std::vector<Body<D>*> right_bodies(bodies.begin() + mid, bodies.end());
    
    return {left_bodies, right_bodies};
}

// Build BVH tree recursively
template <int D>
std::unique_ptr<BVHNode<D>> BVH<D>::build_tree(std::vector<Body<D>*>& bodies, int depth) {
    if (bodies.empty()) return nullptr;
    
    auto node = std::make_unique<BVHNode<D>>();
    
    // If few enough bodies, make a leaf node
    if (bodies.size() <= static_cast<size_t>(max_bodies_per_leaf)) {
        node->is_leaf = true;
        node->bodies = bodies;
        
        // Calculate bounds
        if (!bodies.empty()) {
            node->min_bound = bodies[0]->position;
            node->max_bound = bodies[0]->position;
            
            for (size_t i = 1; i < bodies.size(); ++i) {
                for (int d = 0; d < D; ++d) {
                    node->min_bound[d] = std::min(node->min_bound[d], bodies[i]->position[d]);
                    node->max_bound[d] = std::max(node->max_bound[d], bodies[i]->position[d]);
                }
            }
        }
        
        return node;
    }
    
    // Split bodies in half along the longest axis
    auto [left_bodies, right_bodies] = split_bodies(bodies);
    
    // Build subtrees
    node->is_leaf = false;
    node->left = build_tree(left_bodies, depth + 1);
    node->right = build_tree(right_bodies, depth + 1);
    
    // Calculate bounds from children
    if (node->left && node->right) {
        for (int d = 0; d < D; ++d) {
            node->min_bound[d] = std::min(node->left->min_bound[d], node->right->min_bound[d]);
            node->max_bound[d] = std::max(node->left->max_bound[d], node->right->max_bound[d]);
        }
    } else if (node->left) {
        node->min_bound = node->left->min_bound;
        node->max_bound = node->left->max_bound;
    } else if (node->right) {
        node->min_bound = node->right->min_bound;
        node->max_bound = node->right->max_bound;
    }
    
    return node;
}

// Calculate forces for all bodies
template <int D>
std::vector<Vector<D>> BVH<D>::calculate_forces(const std::vector<Body<D>>& bodies) const {
    std::vector<Vector<D>> forces(bodies.size(), Vector<D>());
    
    if (!root) return forces;
    
    for (size_t i = 0; i < bodies.size(); ++i) {
        forces[i] = calculate_force(bodies[i], root.get());
    }
    
    return forces;
}

// Calculate force on a single body from BVH
template <int D>
Vector<D> BVH<D>::calculate_force(const Body<D>& body, const BVHNode<D>* node) const {
    if (!node) return Vector<D>();
    
    Vector<D> force;
    
    // Leaf node: Calculate force directly with all bodies
    if (node->is_leaf) {
        for (const Body<D>* other : node->bodies) {
            // Skip if nullptr
            if (!other) continue;
            
            // Skip self-interaction
            bool same_position = true;
            for (int d = 0; d < D; ++d) {
                if (std::abs(body.position[d] - other->position[d]) > 1e-9) {
                    same_position = false;
                    break;
                }
            }
            if (same_position) continue;
            
            // Calculate force
            Vector<D> diff = other->position - body.position;
            double dist_sq = diff.magnitude_squared();
            
            // Avoid division by zero with a safe minimum distance
            if (dist_sq < 1e-9) continue;
            
            double dist = std::sqrt(dist_sq);
            double force_mag = G * body.mass * other->mass / (dist_sq * dist);
            
            force += diff.normalized() * force_mag;
        }
    }
    // Internal node: Use approximation if possible, otherwise recurse
    else {
        // Calculate distance to center of node's bounding box
        Vector<D> center;
        for (int d = 0; d < D; ++d) {
            center[d] = (node->min_bound[d] + node->max_bound[d]) / 2.0;
        }
        
        Vector<D> diff = center - body.position;
        double dist = diff.magnitude();
        
        // Avoid division by zero
        if (dist < 1e-9) {
            // If too close to center, recurse to children
            if (node->left) force += calculate_force(body, node->left.get());
            if (node->right) force += calculate_force(body, node->right.get());
            return force;
        }
        
        // Calculate node size
        double node_size = 0;
        for (int d = 0; d < D; ++d) {
            node_size = std::max(node_size, node->max_bound[d] - node->min_bound[d]);
        }
        
        // If node is far enough away, use Barnes-Hut approximation
        // Using global BARNES_HUT_THETA constant
        if (node_size / dist < BARNES_HUT_THETA) {
            // Calculate total mass and center of mass of node
            double total_mass = 0;
            Vector<D> com;
            
            auto process_bodies = [&](const auto& bodies_list) {
                for (const auto& b : bodies_list) {
                    if (!b) continue; // Skip null pointers
                    total_mass += b->mass;
                    for (int d = 0; d < D; ++d) {
                        com[d] += b->position[d] * b->mass;
                    }
                }
            };
            
            // Process bodies in this node and its children
            if (node->left && node->left->is_leaf) process_bodies(node->left->bodies);
            if (node->right && node->right->is_leaf) process_bodies(node->right->bodies);
            
            // Normalize center of mass
            if (total_mass > 1e-9) {
                for (int d = 0; d < D; ++d) {
                    com[d] /= total_mass;
                }
                
                // Recalculate with actual center of mass
                diff = com - body.position;
                dist = diff.magnitude();
                if (dist < 1e-9) return force; // Avoid division by zero
                
                double dist_sq = dist * dist;
                double force_mag = G * body.mass * total_mass / (dist_sq * dist);
                
                force += diff.normalized() * force_mag;
            }
        }
        // Otherwise, recurse into children
        else {
            if (node->left) force += calculate_force(body, node->left.get());
            if (node->right) force += calculate_force(body, node->right.get());
        }
    }
    
    return force;
}

#endif // BVH_TPP
