#ifndef FMM_PARLAY_H
#define FMM_PARLAY_H

#include <vector>
#include <memory>
#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include "vector.h"
#include "body.h"
#include "fmm.h"  // Include base FMM implementation

// ParlayLib-optimized Fast Multipole Method for n-body simulations
template <int D>
class FMM_Parlay {
public:
    FMM_Parlay(const parlay::sequence<Body<D>>& bodies, int max_bodies, int max_lvl, int p);
    
    // Build tree and interaction lists
    void build_tree(const parlay::sequence<Body<D>>& bodies);
    void build_level_lists();
    void build_interaction_lists();
    
    // M2L kernels with different strategies for different sizes
    void m2l_phase();
    void m2l_adaptive(FMMNode<D>* node, parlay::sequence<FMMNode<D>*>& interaction_list);
    void m2l_sequential(FMMNode<D>* node, parlay::sequence<FMMNode<D>*>& interaction_list);
    void m2l_parallel(FMMNode<D>* node, parlay::sequence<FMMNode<D>*>& interaction_list);
    
    // Compute forces
    parlay::sequence<Vector<D>> calculate_forces(const parlay::sequence<Body<D>>& bodies);
    
    // FMM phases
    void p2m_phase();
    
    // M2M phase: translate multipole expansions up the tree
    void m2m_phase();
    void l2l_phase();
    
    // L2P phase: evaluate local expansions at particle positions
    void l2p_phase(parlay::sequence<Vector<D>>& forces, const parlay::sequence<Body<D>>& bodies);
    
    // P2P phase: direct calculation for nearby particles
    void p2p_phase(parlay::sequence<Vector<D>>& forces, const parlay::sequence<Body<D>>& bodies);
    
    // Tuning parameters
    double theta_criterion = 0.5; // More conservative multipole acceptance criterion
    bool use_quadrupole_correction = true; // Enable improved quadrupole terms
    bool adaptive_parallelism = true; // Dynamically choose parallel strategy

private:
    std::unique_ptr<FMMNode<D>> root;
    parlay::sequence<parlay::sequence<FMMNode<D>*>> nodes_by_level;
    int max_bodies_per_leaf;
    int max_level;
    int order;
};

#include "fmm_parlay.cpp"

#endif // FMM_PARLAY_H
