#ifndef FMM_OMP_H
#define FMM_OMP_H

#include <vector>
#include <memory>
#include <omp.h>
#include "vector.h"
#include "body.h"
#include "fmm.h"  // Include base FMM implementation

// OpenMP-optimized Fast Multipole Method for n-body simulations
template <int D>
class FMM_OMP {
private:
    std::unique_ptr<FMMNode<D>> root;
    int max_bodies_per_leaf;
    int max_level;
    int order;
    
    // For level-based parallelism
    std::vector<std::vector<FMMNode<D>*>> nodes_by_level;
    
    // Build list of nodes by level for parallel processing
    void build_level_lists();
    
    // Build interaction lists with more conservative criteria
    void build_interaction_lists();
    
    // P2M phase: compute multipole expansion for leaf nodes
    void p2m_phase();
    
    // M2M phase: translate multipole expansions up the tree
    void m2m_phase();
    
    // M2L phase: translate multipole expansions to local expansions
    void m2l_phase();
    
    // L2L phase: translate local expansions down the tree
    void l2l_phase();
    
    // L2P phase: evaluate local expansions at particle positions
    void l2p_phase(std::vector<Vector<D>>& forces, const std::vector<Body<D>>& bodies);
    
    // P2P phase: direct calculation for nearby particles
    void p2p_phase(std::vector<Vector<D>>& forces, const std::vector<Body<D>>& bodies);

public:
    // Constructor
    FMM_OMP(const std::vector<Body<D>>& bodies, 
           int max_bodies = 10, 
           int max_lvl = 10, 
           int p = 10);
    
    // Calculate forces using optimized OpenMP FMM
    std::vector<Vector<D>> calculate_forces(const std::vector<Body<D>>& bodies);
};

#include "fmm_omp.tpp"

#endif // FMM_OMP_H
