#include <iostream>
#include <vector>
#include <chrono>
#include "utils.h"
#include "methods.h"
using namespace std;

int main(int argc, char* argv[]) {
    int N = argc > 1 ? stoi(argv[1]) : 1e5;
    vector<Body> bodies = generate_random_bodies(N);
    int print_total = 3;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    vector<Vector2D> forces;
    cout << "Calculate gravitational forces between " << N << " random bodies:" << endl;
    if (N <= 5e5) {
        cout << "Brute force O(n^2) sequential approach:" << endl;
        start = std::chrono::high_resolution_clock::now();
        forces = brute_force_seq_n_body(bodies);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        cout << "Time taken: " << duration.count() / 1e6 << " s" << endl;
        for (int i = 0; i < N; i++) {
            if (print_total && (i + 1) % (N / print_total) == 0) {
                cout << "Body #" << i + 1 << " force: (" << forces[i].x << ", " << forces[i].y << ")" << endl;
            }
        }
        cout << endl;
    }
    cout << "Brute force OpenMP parallel approach (memory-intensive):" << endl;
    cout << "Using " << omp_get_max_threads() << " threads..." << endl;
    start = std::chrono::high_resolution_clock::now();
    forces = brute_force_omp_n_body_1(bodies);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "Time taken: " << duration.count() / 1e6 << " s" << endl;
    for (int i = 0; i < N; i++) {
        if (print_total && (i + 1) % (N / print_total) == 0) {
            cout << "Body #" << i + 1 << " force: (" << forces[i].x << ", " << forces[i].y << ")" << endl;
        }
    }
    cout << endl;
    cout << "Brute force OpenMP parallel approach (memory-efficient):" << endl;
    cout << "Using " << omp_get_max_threads() << " threads..." << endl;
    start = std::chrono::high_resolution_clock::now();
    forces = brute_force_omp_n_body_2(bodies);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "Time taken: " << duration.count() / 1e6 << " s" << endl;
    for (int i = 0; i < N; i++) {
        if (print_total && (i + 1) % (N / print_total) == 0) {
            cout << "Body #" << i + 1 << " force: (" << forces[i].x << ", " << forces[i].y << ")" << endl;
        }
    }
    cout << endl;
    cout << "Brute force ParlayLib parallel approach (memory-inefficient):" << endl;
    cout << "Using " << parlay::num_workers() << " workers..." << endl;
    start = std::chrono::high_resolution_clock::now();
    parlay::sequence<Body> bodies_parlay = parlay::to_sequence(bodies);
    parlay::sequence<Vector2D> forces_parlay = brute_force_parlay_n_body_1(bodies_parlay);
    forces = vector<Vector2D>(forces_parlay.begin(), forces_parlay.end());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "Time taken: " << duration.count() / 1e6 << " s" << endl;
    for (int i = 0; i < N; i++) {
        if (print_total && (i + 1) % (N / print_total) == 0) {
            cout << "Body #" << i + 1 << " force: (" << forces[i].x << ", " << forces[i].y << ")" << endl;
        }
    }
    cout << endl;
    cout << "Brute force ParlayLib parallel approach (memory-efficient):" << endl;
    cout << "Using " << parlay::num_workers() << " workers..." << endl;
    start = std::chrono::high_resolution_clock::now();
    bodies_parlay = parlay::to_sequence(bodies);
    forces_parlay = brute_force_parlay_n_body_2(bodies_parlay);
    forces = vector<Vector2D>(forces_parlay.begin(), forces_parlay.end());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "Time taken: " << duration.count() / 1e6 << " s" << endl;
    for (int i = 0; i < N; i++) {
        if (print_total && (i + 1) % (N / print_total) == 0) {
            cout << "Body #" << i + 1 << " force: (" << forces[i].x << ", " << forces[i].y << ")" << endl;
        }
    }
    return 0;
}