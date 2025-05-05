#ifndef OCTREE_TPP
#define OCTREE_TPP

#include <cmath>
#include <algorithm>
#include "octree.h"
#include "utils.h" // Include utils.h for G

template <int D>
OctreeNode<D>::OctreeNode(const Vector<D>& c, double hs)
    : center(c), half_size(hs), total_mass(0), com(), body(nullptr) {
    for (int i = 0; i < (1 << D); ++i) {
        children[i] = nullptr;
    }
}

template <int D>
bool OctreeNode<D>::is_leaf() const {
    return body != nullptr || is_empty();
}

template <int D>
bool OctreeNode<D>::is_empty() const {
    return total_mass == 0;
}

template <int D>
int OctreeNode<D>::get_octant(const Vector<D>& pos) const {
    int octant = 0;
    for (int d = 0; d < D; ++d) {
        if (pos[d] >= center[d]) {
            octant |= (1 << d);
        }
    }
    return octant;
}

template <int D>
void OctreeNode<D>::insert(Body<D>* b) {
    // If node is empty, assign the body directly
    if (is_empty()) {
        body = b;
        total_mass = b->mass;
        com = b->position;
        return;
    }
    
    // If this is an internal node or will become one
    if (!is_leaf() || body != nullptr) {
        // If it was a leaf with one body, move that body to a child
        if (body != nullptr) {
            Body<D>* old_body = body;
            body = nullptr;
            
            int old_octant = get_octant(old_body->position);
            if (!children[old_octant]) {
                Vector<D> new_center = center;
                for (int d = 0; d < D; ++d) {
                    if (old_octant & (1 << d)) {
                        new_center[d] += half_size / 2;
                    } else {
                        new_center[d] -= half_size / 2;
                    }
                }
                children[old_octant] = std::make_unique<OctreeNode<D>>(new_center, half_size / 2);
            }
            
            children[old_octant]->insert(old_body);
        }
        
        // Now insert the new body
        int octant = get_octant(b->position);
        if (!children[octant]) {
            Vector<D> new_center = center;
            for (int d = 0; d < D; ++d) {
                if (octant & (1 << d)) {
                    new_center[d] += half_size / 2;
                } else {
                    new_center[d] -= half_size / 2;
                }
            }
            children[octant] = std::make_unique<OctreeNode<D>>(new_center, half_size / 2);
        }
        
        children[octant]->insert(b);
        
        // Update mass and center of mass
        double new_total_mass = total_mass + b->mass;
        for (int d = 0; d < D; ++d) {
            com[d] = (com[d] * total_mass + b->position[d] * b->mass) / new_total_mass;
        }
        total_mass = new_total_mass;
    }
}

template <int D>
Vector<D> OctreeNode<D>::calculate_force(const Body<D>& b, double theta) const {
    Vector<D> force;
    
    if (is_empty()) {
        return force;
    }
    
    // If this is a leaf with a body, and it's not the same body we're calculating force for
    if (is_leaf() && body) {
        // Use a more precise comparison with tolerance
        bool same_position = true;
        for (int d = 0; d < D; ++d) {
            if (std::abs(b.position[d] - body->position[d]) > 1e-9) {
                same_position = false;
                break;
            }
        }
        
        if (!same_position) {
            Vector<D> diff = body->position - b.position;
            double dist_sq = diff.magnitude_squared();
            
            // Use a smaller tolerance for numerical stability
            if (dist_sq < 1e-9) return force; // Avoid division by zero
            
            double dist = std::sqrt(dist_sq);
            double force_mag = G * b.mass * body->mass / (dist_sq * dist);
            
            return diff.normalized() * force_mag;
        }
    }
    
    // If not leaf, check if we can use approximation with the specified theta parameter
    if (!is_leaf()) {
        Vector<D> diff = com - b.position;
        double dist = diff.magnitude();
        
        // Avoid division by zero
        if (dist < 1e-9) {
            // If too close to center, recurse to children
            for (int i = 0; i < (1 << D); ++i) {
                if (children[i]) {
                    force += children[i]->calculate_force(b, theta);
                }
            }
            return force;
        }
        
        // If ratio of size to distance is small enough, use approximation (Barnes-Hut criterion)
        if (2.0 * half_size / dist < theta) {
            double dist_sq = dist * dist;
            double force_mag = G * b.mass * total_mass / (dist_sq * dist);
            
            return diff.normalized() * force_mag;
        }
        
        // Otherwise, recurse into children
        for (int i = 0; i < (1 << D); ++i) {
            if (children[i]) {
                force += children[i]->calculate_force(b, theta);
            }
        }
    }
    
    return force;
}

template <int D>
Octree<D>::Octree(const std::vector<Body<D>>& bodies) {
    if (bodies.empty()) {
        return;
    }
    
    // Find bounds of all bodies
    Vector<D> min_pos = bodies[0].position;
    Vector<D> max_pos = bodies[0].position;
    
    for (const auto& body : bodies) {
        for (int d = 0; d < D; ++d) {
            min_pos[d] = std::min(min_pos[d], body.position[d]);
            max_pos[d] = std::max(max_pos[d], body.position[d]);
        }
    }
    
    // Calculate center and half size for the root node
    Vector<D> center;
    double max_half_size = 0;
    
    for (int d = 0; d < D; ++d) {
        center[d] = (min_pos[d] + max_pos[d]) / 2;
        max_half_size = std::max(max_half_size, std::abs(max_pos[d] - min_pos[d]) / 2);
    }
    
    // Add a bit of padding
    max_half_size *= 1.01;
    
    // Create root node
    root = std::make_unique<OctreeNode<D>>(center, max_half_size);
    
    // Insert all bodies
    for (size_t i = 0; i < bodies.size(); ++i) {
        // We need to insert pointers, but can't use address of vector elements
        // as they might be invalidated during vector resizing
        // This is a workaround for the sample code
        Body<D>* body_ptr = const_cast<Body<D>*>(&bodies[i]);
        root->insert(body_ptr);
    }
}

template <int D>
std::vector<Vector<D>> Octree<D>::calculate_forces(const std::vector<Body<D>>& bodies) const {
    std::vector<Vector<D>> forces(bodies.size());
    
    if (!root) {
        return forces;
    }
    
    for (size_t i = 0; i < bodies.size(); ++i) {
        forces[i] = root->calculate_force(bodies[i], BARNES_HUT_THETA);  // Use global constant
    }
    
    return forces;
}

#endif // OCTREE_TPP
