#ifndef METHODS_H
#define METHODS_H

#include <vector>
#include <omp.h>
#include "body.h"
#include "vector.h"
#include "utils.h"  // Include utils.h first for common utility functions

// Forward declare classes to avoid circular dependencies
template <int D> class Octree;
template <int D> class BVH;
template <int D> class FMM;

// Include implementation headers after forward declarations
#include "octree.h"
#include "bvh.h"
#include "fmm.h"
#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

// Brute force implementations
template <int D>
std::vector<Vector<D>> brute_force_seq_n_body(const std::vector<Body<D>>& bodies);

template <int D>
std::vector<Vector<D>> brute_force_omp_n_body_1(const std::vector<Body<D>>& bodies);

template <int D>
std::vector<Vector<D>> brute_force_omp_n_body_2(const std::vector<Body<D>>& bodies);

template <int D>
parlay::sequence<Vector<D>> brute_force_parlay_n_body_1(const parlay::sequence<Body<D>>& bodies);

template <int D>
parlay::sequence<Vector<D>> brute_force_parlay_n_body_2(const parlay::sequence<Body<D>>& bodies);

// Barnes-Hut implementations
template <int D>
std::vector<Vector<D>> barnes_hut_seq_n_body(const std::vector<Body<D>>& bodies, double theta = 0.5);

template <int D>
std::vector<Vector<D>> barnes_hut_omp_n_body(const std::vector<Body<D>>& bodies, double theta = 0.5);

template <int D>
parlay::sequence<Vector<D>> barnes_hut_parlay_n_body(const parlay::sequence<Body<D>>& bodies, double theta = 0.5);

// BVH implementations
template <int D> 
std::vector<Vector<D>> bvh_seq_n_body(const std::vector<Body<D>>& bodies, int max_bodies_per_leaf = 16);

template <int D>
std::vector<Vector<D>> bvh_omp_n_body(const std::vector<Body<D>>& bodies, int max_bodies_per_leaf = 16);

template <int D>
parlay::sequence<Vector<D>> bvh_parlay_n_body(const parlay::sequence<Body<D>>& bodies, int max_bodies_per_leaf = 16);

// FMM implementations
template <int D>
std::vector<Vector<D>> fmm_seq_n_body(const std::vector<Body<D>>& bodies, 
                                     int max_bodies_per_leaf = 64, 
                                     int max_level = 6, 
                                     int order = 10);

template <int D>
std::vector<Vector<D>> fmm_omp_n_body(const std::vector<Body<D>>& bodies, 
                                     int max_bodies_per_leaf = 64, 
                                     int max_level = 6, 
                                     int order = 10);

template <int D>
parlay::sequence<Vector<D>> fmm_parlay_n_body(const parlay::sequence<Body<D>>& bodies,
                                            int max_bodies_per_leaf = 64, 
                                            int max_level = 6, 
                                            int order = 10);

// Helpers to update positions and velocities
template <int D>
void update_body_velocities(std::vector<Body<D>>& bodies, 
                           const std::vector<Vector<D>>& forces,
                           double dt);

template <int D>
void update_body_positions(std::vector<Body<D>>& bodies, double dt);

#endif // METHODS_H