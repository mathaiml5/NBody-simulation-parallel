#include "methods.h"
using namespace std;

// Brute force O(n^2) sequential approach
vector<Vector2D> brute_force_seq_n_body(const vector<Body>& bodies) {
    int N = bodies.size();
    vector<Vector2D> forces(N);
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            // Compute values
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dist_sq = dx * dx + dy * dy;
            if (dist_sq == 0) continue;
            double dist_cb = dist_sq * sqrt(dist_sq);
            // Calculate forces
            double f_x = bodies[i].mass * bodies[j].mass * dx / dist_cb;
            double f_y = bodies[i].mass * bodies[j].mass * dy / dist_cb;
            forces[j].x += f_x;
            forces[j].y += f_y;
            forces[i].x -= f_x;
            forces[i].y -= f_y;
        }
    }
    // Apply gravitational constant
    for (int i = 0; i < N; i++) {
        forces[i].x *= grav;
        forces[i].y *= grav;
    }
    return forces;
}

// Brute force O(n^2) work parallel approach using OpenMP (memory-intensive)
vector<Vector2D> brute_force_omp_n_body_1(const vector<Body>& bodies) {
    int N = bodies.size();
    vector<Vector2D> forces(N);
    int num_threads = omp_get_max_threads();
    vector<vector<Vector2D>> local(num_threads, vector<Vector2D>(N));
    // Compute iterations of outer loop in parallel
    // Threads work on local copies
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                // Compute values
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dist_sq = dx * dx + dy * dy;
                if (dist_sq == 0) continue;
                double dist_cb = dist_sq * sqrt(dist_sq);
                // Calculate forces
                double f_x = bodies[i].mass * bodies[j].mass * dx / dist_cb;
                double f_y = bodies[i].mass * bodies[j].mass * dy / dist_cb;
                local[t][j].x += f_x;
                local[t][j].y += f_y;
                local[t][i].x -= f_x;
                local[t][i].y -= f_y;
            }
        }
    }
    // Combine forces from all threads
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < N; i++) {
            forces[i].x += local[t][i].x;
            forces[i].y += local[t][i].y;
        }
    }
    // Apply gravitational constant
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        forces[i].x *= grav;
        forces[i].y *= grav;
    }
    return forces;
}

// Brute force O(n^2) work parallel approach using OpenMP (memory-efficient)
vector<Vector2D> brute_force_omp_n_body_2(const vector<Body>& bodies) {
    int N = bodies.size();
    vector<Vector2D> forces(N);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            // Compute values
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dist_sq = dx * dx + dy * dy;
            if (dist_sq == 0) continue;
            double dist_cb = dist_sq * sqrt(dist_sq);
            // Calculate forces
            double f_x = bodies[i].mass * bodies[j].mass * dx / dist_cb;
            double f_y = bodies[i].mass * bodies[j].mass * dy / dist_cb;
            forces[i].x -= f_x;
            forces[i].y -= f_y;
        }
    }
    // Apply gravitational constant
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        forces[i].x *= grav;
        forces[i].y *= grav;
    }
    return forces;
}

// Brute force O(n^2) work parallel approach using ParlayLib (memory-intensive)
parlay::sequence<Vector2D> brute_force_parlay_n_body_1(const parlay::sequence<Body>& bodies) {
    int N = bodies.size();
    int num_workers = parlay::num_workers();
    // Thread-local storage
    auto local = parlay::tabulate(num_workers, [N](size_t) {
        return parlay::sequence<Vector2D>(N);
    });
    size_t grain_size = std::max<size_t>(1, N / (4 * num_workers));
    parlay::parallel_for(0, N, [&](int i) {
        size_t worker_id = parlay::worker_id();
        for (int j = i + 1; j < N; j++) {
            // Compute values
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dist_sq = dx * dx + dy * dy;
            if (dist_sq == 0) continue;
            double dist_cb = dist_sq * sqrt(dist_sq);
            // Calculate forces
            double f_x = bodies[i].mass * bodies[j].mass * dx / dist_cb;
            double f_y = bodies[i].mass * bodies[j].mass * dy / dist_cb;
            local[worker_id][j].x += f_x;
            local[worker_id][j].y += f_y;
            local[worker_id][i].x -= f_x;
            local[worker_id][i].y -= f_y;
        }
    }, grain_size);
    // Combine forces from all threads
    auto forces = parlay::sequence<Vector2D>(N);
    for (int t = 0; t < num_workers; t++) {
        for (int i = 0; i < N; i++) {
            forces[i].x += local[t][i].x;
            forces[i].y += local[t][i].y;
        }
    }
    // Apply gravitational constant
    parlay::parallel_for(0, N, [&](size_t i) {
        forces[i].x *= grav;
        forces[i].y *= grav;
    });
    return forces;
}


// Brute force O(n^2) work parallel approach using ParlayLib (memory-efficient)
parlay::sequence<Vector2D> brute_force_parlay_n_body_2(const parlay::sequence<Body>& bodies) {
    int N = bodies.size();
    auto forces = parlay::sequence<Vector2D>(N);
    int num_workers = parlay::num_workers();
    size_t grain_size = std::max<size_t>(1, N / (4 * num_workers));
    parlay::parallel_for(0, N, [&](int i) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            // Compute values
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dist_sq = dx * dx + dy * dy;
            if (dist_sq == 0) continue;
            double dist_cb = dist_sq * sqrt(dist_sq);
            // Calculate forces
            double f_x = bodies[i].mass * bodies[j].mass * dx / dist_cb;
            double f_y = bodies[i].mass * bodies[j].mass * dy / dist_cb;
            forces[i].x -= f_x;
            forces[i].y -= f_y;
        }
    }, grain_size);
    // Apply gravitational constant
    parlay::parallel_for(0, N, [&](size_t i) {
        forces[i].x *= grav;
        forces[i].y *= grav;
    });
    return forces;
}