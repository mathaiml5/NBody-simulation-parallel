#ifndef FMM_H
#define FMM_H

#include <vector>
#include <complex>
#include <memory>
#include <array>

#include "vector.h"
#include "body.h"
#include "utils.h" // Include utils.h for G, complex math helpers, and other constants

// Forward declarations
template <int D> class FMMNode;
template <int D> class FMM;

// Multipole and Local expansion coefficients
template <int D, int P>
struct Expansion {
    // For 2D: Complex coefficients
    // For 3D: Spherical harmonic coefficients
    std::vector<std::complex<double>> coeff;

    Expansion() {
        if constexpr (D == 2) {
            // 2D: P complex coefficients
            coeff.resize(P + 1); // Include 0th term
        } else {
            // 3D: P^2 coefficients for spherical harmonics
            coeff.resize((P + 1) * (P + 1));
        }
        clear();
    }

    // Reset expansion to zero
    void clear() {
        std::fill(coeff.begin(), coeff.end(), std::complex<double>(0.0, 0.0));
    }

    // Add another expansion to this one
    Expansion& operator+=(const Expansion& other) {
        for (size_t i = 0; i < coeff.size(); ++i) {
            coeff[i] += other.coeff[i];
        }
        return *this;
    }
};

// FMM Quadtree/Octree node
template <int D>
class FMMNode {
public:
    // Bounds information
    Vector<D> center;
    double half_size;

    // Bodies contained in this node
    std::vector<Body<D>*> bodies;

    // Multipole and local expansions
    Expansion<D, 10> multipole;  // Using order 10 for demonstration
    Expansion<D, 10> local;

    // Child nodes (4 for 2D, 8 for 3D)
    std::array<std::unique_ptr<FMMNode>, 1 << D> children;

    // Parent node
    FMMNode* parent;

    // Node level in the tree
    int level;

    // Interaction list (nodes in M2L interactions)
    std::vector<FMMNode*> interaction_list;

    // Neighbor list (nodes for direct calculations)
    std::vector<FMMNode*> neighbor_list;

    // Constructor
    FMMNode(const Vector<D>& c, double hs, FMMNode* p = nullptr, int lvl = 0);

    // Insert a body into the node
    void insert(Body<D>* body, int max_bodies, int max_level);

    // Check if this is a leaf node
    bool is_leaf() const;

    // Compute multipole expansion from bodies (P2M)
    void compute_multipole(int order);

    // Translate multipole expansion from children to parent (M2M)
    void translate_multipole_to_parent(int order);

    // Translate multipole expansion to local expansion (M2L)
    void translate_multipole_to_local(FMMNode<D>* source, int order);

    // Translate parent's local expansion to children (L2L)
    void translate_local_to_children(int order);

    // Compute direct forces between bodies (P2P)
    void compute_direct_forces(std::vector<Vector<D>>& forces, 
                               const std::vector<Body<D>>& all_bodies);

    // Evaluate local expansion at body positions (L2P)
    void evaluate_local_expansion(std::vector<Vector<D>>& forces, 
                                  const std::vector<Body<D>>& all_bodies,
                                  int order);
};

// Main FMM class
template <int D>
class FMM {
public:
    std::unique_ptr<FMMNode<D>> root;
    int max_bodies_per_leaf;
    int max_level;
    int order;

    // Constructor
    FMM(const std::vector<Body<D>>& bodies, 
        int max_bodies = 10, 
        int max_lvl = 10, 
        int p = 10);
    
    // Build the FMM tree
    void build_tree(const std::vector<Body<D>>& bodies);
    
    // Build interaction lists for M2L and direct calculation
    void build_interaction_lists();
    
    // Calculate forces using FMM
    std::vector<Vector<D>> calculate_forces(const std::vector<Body<D>>& bodies);
    
    // The upward pass: P2M and M2M
    void upward_pass();
    
    // The interaction pass: M2L
    void interaction_pass();
    
    // The downward pass: L2L and L2P
    void downward_pass(std::vector<Vector<D>>& forces, const std::vector<Body<D>>& bodies);
    
    // Calculate force on a single body using improved accuracy method
    Vector<D> calculate_accurate_force(const Body<D>& body);

private:
    // Recursively build interaction lists
    void build_interaction_lists_recursive(FMMNode<D>* node);

    // Execute the FMM algorithm
    void execute_fmm();
    
    // Build the tree structure
    std::unique_ptr<FMMNode<D>> build_tree_recursive(
        const Vector<D>& center, 
        double half_size, 
        std::vector<Body<D>*>& bodies_subset, 
        FMMNode<D>* parent, 
        int level);
};

// Type aliases
using FMM2D = FMM<2>;
using FMM3D = FMM<3>;

// Include implementation
#include "fmm.cpp"

#endif // FMM_H
