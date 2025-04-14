#ifndef BASELINE_H
#define BASELINE_H

#include <cmath>
#include <vector>
#include <omp.h>
#include "utils.h"
#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"

std::vector<Vector2D> brute_force_seq_n_body(const std::vector<Body>& bodies);

std::vector<Vector2D> brute_force_omp_n_body(const std::vector<Body>& bodies);

parlay::sequence<Vector2D> brute_force_parlay_n_body(const parlay::sequence<Body>& bodies);

#endif