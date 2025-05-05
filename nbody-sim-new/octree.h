#ifndef OCTREE_H
#define OCTREE_H

#include <vector>
#include <memory>
#include "vector.h"
#include "body.h"
#include "utils.h" // Include utils.h for G

// This class represents a node in the Barnes-Hut octree
template <int D>
class OctreeNode {
public:
    // Bounds of this node
    Vector<D> center;      // Center of the cell
    double half_size;      // Half the width of the cell
    
    // Body or center of mass information
    double total_mass;     // Total mass of bodies in this cell
    Vector<D> com;         // Center of mass of bodies in this cell
    
    // For leaf nodes with a single body
    Body<D>* body;         // Direct reference to a body (nullptr if not leaf)
    
    // Child nodes
    std::unique_ptr<OctreeNode<D>> children[1 << D]; // 4 for 2D, 8 for 3D
    
    OctreeNode(const Vector<D>& c, double hs);
    
    // Insert a body into this node, possibly creating child nodes
    void insert(Body<D>* b);
    
    // Calculate force on a body
    Vector<D> calculate_force(const Body<D>& b, double theta) const;
    
    // Determine which octant a point belongs to
    int get_octant(const Vector<D>& pos) const;
    
    // Check if node is leaf
    bool is_leaf() const;
    
    // Check if node is empty
    bool is_empty() const;
};

// Type aliases
using OctreeNode2D = OctreeNode<2>;
using OctreeNode3D = OctreeNode<3>;

// Main Octree class to build and manage the tree
template <int D>
class Octree {
public:
    std::unique_ptr<OctreeNode<D>> root;

    // Constructor - removed theta_param
    Octree(const std::vector<Body<D>>& bodies);
    
    // Calculate forces on all bodies
    std::vector<Vector<D>> calculate_forces(const std::vector<Body<D>>& bodies) const;
};

// Type aliases
using Octree2D = Octree<2>;
using Octree3D = Octree<3>;

// Include implementation
#include "octree.cpp"

#endif // OCTREE_H
