# N-body Simulation Benchmark Suite

This project implements and benchmarks various N-body simulation algorithms with sequential and parallel implementations in C++ and using CUDA. The simulation calculates gravitational forces between N bodies in either 2D or 3D space. Scripts to gather runtime performance metrics and analyze results and provide visualizations are provided.

## Algorithms Implemented

The suite contains four main algorithm categories, each with sequential and parallel implementations:

1. **Brute Force O(N²)** - Direct calculation of all pairwise interactions
2. **Barnes-Hut** - Tree-based hierarchical algorithm using spatial partitioning
3. **BVH (Bounding Volume Hierarchy)** - Space partitioning using Hilbert curve ordering
4. **FMM (Fast Multipole Method)** - Hierarchical algorithm with multipole expansions

## Design and Optimizations

### Brute Force Methods

The naive approach calculates all pairwise interactions between bodies with O(N²) complexity.

- **Sequential**: Simple nested loops over all body pairs
- **OpenMP (memory-intensive)**: Parallelizes outer loop with `#pragma omp parallel for` and pre-allocates result array
- **OpenMP (memory-efficient)**: Uses thread-local storage to reduce memory contention
- **ParlayLib (memory-intensive)**: Uses `parlay::parallel_for` with full result array
- **ParlayLib (memory-efficient)**: Task-based batching with reduced memory footprint

### Barnes-Hut Algorithm (O(N log N))

Uses an octree (quadtree in 2D) to approximate distant interactions.

**Key steps:**
1. **Tree Construction**: Recursively subdivides space into cells
2. **Calculate CoM (Center of Mass)**: Bottom-up pass to compute cell properties
3. **Force Calculation**: Traverses the tree to compute forces

**Parallelism:**
- **OpenMP**: Parallelizes tree traversal during force calculation phase
- **ParlayLib**: Uses parallel tree construction and force calculation with work-stealing

### BVH (Bounding Volume Hierarchy) (O(N log N))

Uses a binary tree based on Hilbert curve ordering for better cache coherence.

**Key steps:**
1. **Object Sorting**: Bodies are sorted along a Hilbert space-filling curve
2. **BVH Construction**: Builds a binary tree with spatial hierarchy
3. **Force Calculation**: Traverses the tree to compute forces

**Parallelism:**
- **OpenMP**: Parallel sorting and force calculation
- **ParlayLib**: Parallel construction and traversal with improved load balancing

### FMM (Fast Multipole Method) (O(N))

Uses multipole expansions to approximate interactions between distant clusters.

**Key steps:**
1. **Tree Construction**: Creates a spatial hierarchy similar to Barnes-Hut
2. **Upward Pass (P2M, M2M)**: Computes multipole expansions bottom-up
3. **Transfer Pass (M2L)**: Translates multipoles to local expansions
4. **Downward Pass (L2L, L2P)**: Evaluates local expansions at body positions
5. **Direct Pass (P2P)**: Calculates near-field interactions directly

**Parallelism:**
- **OpenMP**: Parallel P2M, M2L, and P2P phases with thread synchronization
- **ParlayLib**: All phases parallelized with dependency tracking for optimal work distribution
- Each phase of FMM leverages parallelism:
    - P2M (Particle to Multipole): At the finest level (leaf nodes of the tree), this phase computes the multipole expansion for each leaf box based on all bodies contained within it. The computation for each leaf box is independent of other leaf boxes. A #pragma omp parallel for directive can be used to parallelize the loop that iterates over all leaf boxes, with each thread computing the multipole expansion for a subset of leaf boxes. For parlaylib, it maps well to parlay::parallel_for to parallelize the loop over particles within a leaf box or over all leaf boxes. 
    - M2M (Multipole to Multipole): In this upward pass through the tree, the multipole expansion of a child box is translated to the center of its parent box and added to the parent's multipole expansion. This process is done level by level, from the leaf levels up to the root. The M2M translation for a parent box depends on the completed M2M computations of its children. Parallelism can be exploited at each tree level. For a given level, the M2M computations for different parent boxes at that level are independent. A #pragma omp parallel for can be used to parallelize the loop over boxes at each level (starting from the level above the finest) in the upward pass. Alternatively, a task-based approach can be used, where a task is created for the M2M operation of a box, and this task depends on the completion of tasks for its children. #pragma omp taskgroup can be used to ensure all child tasks complete before the parent task proceeds. In the case of parlaylib, hierarchical dependency and independent sibling work can be parallelized using parlay::spawn and parlay::sync within a recursive function, often structured with parlay::par_do to spawn parallel tasks for children's upward passes.
    - M2L (Multipole to Local): This is the most computationally expensive portion of the FMM and involves interactions between well-separated boxes. For a given box, the multipole expansions of boxes in its interaction list (boxes that are far enough away but not children of its parent or the parent's siblings, etc.) are translated into a local expansion at the center of the given box. Since the M2L computation for a box is independent of the M2L computations for other boxes at the same level and the interactions between a box and its interaction list can both be parallelized. A combination of #pragma omp parallel for iterating over boxes at a given level and potentially nested #pragma omp for or tasks for processing the interaction list of each box can be used. Due to the irregular nature of interaction lists in adaptive FMM, a task-based approach where each task handles the M2L contribution from one box in the interaction list to the target box, or even a task per target box that iterates over its interaction list, can help with load balancing. For parlaylib, this phase can be parallelized by iterating over the target boxes at a given level using parlay::parallel_for, and/or by using parlay::parallel_for or parlay::spawn within a parlay::taskgroup to parallelize the interactions with the boxes in the interaction list of a given target box.
    - L2L (Local to Local): In this downward pass through the tree, the local expansion at the center of a parent box is translated to the center of its child boxes and added to the child's local expansion at each level, from root to leaves. Similar to M2M, the L2L translation for a child box depends on the parent's local expansion. Parallelism can be achieved by parallelizing the loop over boxes at each level in the downward pass (starting from the level below the root). #pragma omp parallel for can be used. A task-based approach, where a task is created for the L2L operation of a parent box that spawns tasks for its children, can also be effective. The L2L translations from a parent to its different children are independent given the parent's local expansion. In parlaylib, we can use parlay::parallel_for to translate the parent's local expansion to all its children at once, followed by recursive calls to children's downward passes, potentially initiated using parlay::spawn within parlay::par_do.
    - L2P (Local to Particle): At each leaf node (representing  one or more bodies), this phase evaluates the effect of the local expansion at the center of the leaf box on the particles within that box. Since this calculation for bodies within a leaf box is independent of the computation for particles in other leaf boxes and that for each body in a box are independent, this can be efficiently parallelized. A #pragma omp parallel for can be used to parallelize the loop over leaf boxes, and potentially another nested #pragma omp for to parallelize the loop over particles within each box. In the case of parlaylib we can directly to using parlay::parallel_for to parallelize the loop over particles in each leaf box.
    - P2P (Particle to Particle): This phase computes the direct interactions between particles in nearby boxes that are not well-separated enough to be handled by the M2L approximation. such as bodies within a leaf node box and particles in its neighboring leaf node boxes (including itself). The P2P computation between pairs of nearby boxes is generally independent. The computation of interactions between particles within these pairs of boxes is also highly parallel. This phase is well-suited for #pragma omp parallel for to iterate over the pairs of interacting boxes or to iterate over the particles and compute their direct interactions with nearby particles. Efficiently managing the neighbor lists for P2P interactions is crucial for performance. In the case of parlaylib, we can use parlay::parallel_for over particles within a box or over the pairs of interacting nearby boxes with thread-safe accumulation of potential on particles as a key consideration.

- Other Parallelism notes
    - Using task for Tree Traversal: For the M2M and L2L phases, and parts of the M2L phase, OpenMP tasks can provide a more better load-balanced parallelization. For example, a recursive function can be used for tree traversal and for a given node, tasks can be created for processing its children (for M2M or L2L). #pragma omp task can be placed before the recursive calls to process child nodes and a #pragma omp taskwait can be used after creating tasks for children to ensure that the child computations are complete before the parent computation (M2M) or before proceeding to the next level downwards (L2L). For M2L, tasks can be created for each box in the interaction list of a given box.

## Performance Optimizations

1. **Vector Operations**: We use vector operations for coordinate calculations where possible for efficiency
2. **Numerical Stability**: We check for small values of forces and small values of reciprocals of squared distances that can cause issues such as division by zero and convergence problems and set appropriate limit checks. This is especially critical for accurate multipole expansions which are like Taylor series expansions.

## Building the Project

The project requires a C++17 compliant compiler, OpenMP, and the ParlayLib library.

```bash
# Clone the repository
git clone <repository-url>
cd nbody-sim-new

# Build the project
make clean
make
```

## Running the Benchmark

### Command Line Options

```
Usage: ./nbody_sim [options]
Options:
  -d, --dim <2|3>     Set simulation dimension (default: 3)
  -N, --bodies <num>  Set number of bodies (default: 1000)
  -a, --accuracy <0|1> Enable accuracy calculation (default: 0 - OFF)
  -m, --methods <str> Specify which methods to run (default: all)
                      a=bruteforce, b=barnes-hut, h=hilbert bvh, f=fmm
                      Example: -m bf runs Barnes-Hut and FMM only
  -h, --help          Display this help message
```

### Examples

```bash
# Run all algorithms with 10,000 bodies in 3D
./nbody_sim -N 10000

# Run only Barnes-Hut algorithms with 1 million bodies in 2D
./nbody_sim -N 1000000 -d 2 -m b

# Run both BVH and FMM with accuracy calculation
./nbody_sim -N 100000 -m hf -a 1

# Run brute force methods with 2 million bodies (overriding size limit)
./nbody_sim -N 2000000 -m a
```

## Benchmark Script

The `run_simulations.sh` script automates running benchmarks with various parameters:

```bash
# Run the benchmark script
./run_simulations.sh
```

This script:

1. Builds the executable
2. Runs simulations with increasing body counts (1K to 5M)
3. Tests both 2D and 3D configurations
4. Runs separate tests with and without accuracy calculation
5. Generates CSV and log files in the results directory

## Results Interpretation

Results are saved in two formats:
- `.csv` files containing performance data for analysis
- `.out` files with detailed logs of each run

The CSV format includes:
- Algorithm method
- Number of bodies
- Dimension (2D or 3D) 
- Execution time in seconds
- Accuracy percentage (when enabled)

## Key Parallelization Strategies

### OpenMP Implementation

- **Brute Force**: Simple loop parallelism with `#pragma omp parallel for`
- **Barnes-Hut**: Tree traversal parallelism with shared read-only tree
- **BVH**: Parallelized sorting and force computation phases
- **FMM**: Phase-specific parallelism with barrier synchronization

### ParlayLib Implementation

- **Work-stealing schedulers** for better load balancing
- **Nested parallelism** for hierarchical algorithms
- **Parallel scans and sorts** for efficient tree construction
- **Lock-free data structures** to reduce contention

### Global Config Parameters
Following parameters are provided at the top of `utils.h` to fine tune and adjust performance 
- `const double G = 6.67430e-11;` Gravitational constant: Default value is when using units of kg, m, s for mass, distance, and time. If using other mass units such as solar or earth mass or other distances units such as parsec this constant will need to be adjusted
- `const double BARNES_HUT_THETA = 0.25;` Barnes-Hut approximation parameter: values closer to zero yield better accuracy. 
- `const double EPSILON = 1e-11;`  Small value threshold to avoid division by zero errors
- `const double SOFTENING = 1e-6;`  Softening parameter for close interactions
- `const double ACCURACY_PCT_THRESHOLD = 0.01;`  Threshold % for accuracy: if calculated value from simulation is within 1% of reference value (for example, force from an accurate brute force calculation) it is deemed accurate
- `const double ACCURACY_FORCE_THRESHOLD = 1e-20;` 
Set this threshold below which force values are not considered accurate, otherwise it might lead to instability.

## Known Limitations

- Very large simulations (N > 10M) may require significant memory
- Brute force methods are impractical for N > 1M (can be overridden with `-m a`)
- Accuracy calculations double the memory requirements

## Contributors

- Vishak Srikanth
- Rohan Phanse
- Areeb Gani
