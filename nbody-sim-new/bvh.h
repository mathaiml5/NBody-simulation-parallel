#ifndef BVH_H
#define BVH_H

#include <vector>
#include <memory>
#include <algorithm>

#include "vector.h"
#include "body.h"
#include "utils.h" // Include utils.h for G

// AABB (Axis-Aligned Bounding Box) class
template <int D>
struct AABB {
    Vector<D> min_point;
    Vector<D> max_point;
    
    AABB() : min_point(), max_point() {}
    
    AABB(const Vector<D>& min_p, const Vector<D>& max_p)
        : min_point(min_p), max_point(max_p) {}
    
    // Create AABB for a single body
    explicit AABB(const Body<D>& body) {
        min_point = max_point = body.position;
    }
    
    // Merge two AABBs
    AABB& merge(const AABB& other) {
        for (int d = 0; d < D; ++d) {
            min_point[d] = std::min(min_point[d], other.min_point[d]);
            max_point[d] = std::max(max_point[d], other.max_point[d]);
        }
        return *this;
    }
    
    // Center of the AABB
    Vector<D> center() const {
        Vector<D> c;
        for (int d = 0; d < D; ++d) {
            c[d] = (min_point[d] + max_point[d]) / 2.0;
        }
        return c;
    }
    
    // Half-size of the AABB
    double half_size() const {
        double hs = 0.0;
        for (int d = 0; d < D; ++d) {
            hs = std::max(hs, (max_point[d] - min_point[d]) / 2.0);
        }
        return hs;
    }
    
    // Check if two AABBs overlap
    bool overlaps(const AABB& other) const {
        for (int d = 0; d < D; ++d) {
            if (max_point[d] < other.min_point[d] || min_point[d] > other.max_point[d]) {
                return false;
            }
        }
        return true;
    }
    
    // Check if AABB contains a point
    bool contains(const Vector<D>& point) const {
        for (int d = 0; d < D; ++d) {
            if (point[d] < min_point[d] || point[d] > max_point[d]) {
                return false;
            }
        }
        return true;
    }
};

// BVH Node
template <int D>
class BVHNode {
public:
    Vector<D> min_bound;
    Vector<D> max_bound;
    std::vector<Body<D>*> bodies;  // Change to pointer type
    std::unique_ptr<BVHNode<D>> left;
    std::unique_ptr<BVHNode<D>> right;
    bool is_leaf;  // Make this a variable, not a method
    AABB<D> aabb;  // Add the missing AABB field
    
    BVHNode() : is_leaf(true) {}
};

// BVH Tree for n-body simulation
template <int D>
class BVH {
public:
    std::unique_ptr<BVHNode<D>> root;
    int max_bodies_per_leaf;
    
    // Constructor
    BVH(const std::vector<Body<D>>& bodies, int max_bodies = 16);
    
    // Calculate forces on all bodies
    std::vector<Vector<D>> calculate_forces(const std::vector<Body<D>>& bodies) const;
    
    // Calculate forces between a body and a BVH node
    Vector<D> calculate_force(const Body<D>& body, const BVHNode<D>* node) const;

private:
    // Build a BVH tree from a vector of bodies
    std::unique_ptr<BVHNode<D>> build_tree(std::vector<Body<D>*>& bodies, int depth);
    
    // Split the bodies along the longest axis
    std::pair<std::vector<Body<D>*>, std::vector<Body<D>*>> split_bodies(
        std::vector<Body<D>*>& bodies);
};

// Type aliases
using BVH2D = BVH<2>;
using BVH3D = BVH<3>;

// Include implementation
#include "bvh.cpp"

#endif // BVH_H
