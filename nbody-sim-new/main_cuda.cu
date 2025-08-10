#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "utils.h"
#include <cuda_runtime.h>
using namespace std; 

struct CUDAVector2D {
    float x;
    float y;
    __host__ __device__ CUDAVector2D(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
};

struct CUDABody {
    float x;
    float y;
    float mass;
    float padding;
    __host__ __device__ CUDABody(float x = 0.0f, float y = 0.0f, float mass = 0.0f) : x(x), y(y), mass(mass), padding(0.0f) {}
};

// Built upon: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
__global__ void calculate_forces(CUDABody *bodies, CUDAVector2D *forces, int N) {
    extern __shared__ CUDABody shBodies[];
    CUDABody myBody;
    CUDAVector2D myForce = {0.0, 0.0};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < N) {
        myBody = bodies[gtid];
        for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; tile++) {
            int idx = tile * blockDim.x + threadIdx.x;
            if (idx < N) {
                shBodies[threadIdx.x] = bodies[idx];
            } else {
                shBodies[threadIdx.x] = CUDABody(0.0f, 0.0f, 0.0f);
            }
            __syncthreads();
            #pragma unroll 4
            for (int i = 0; i < blockDim.x; i++) {
                int j = tile * blockDim.x + i;
                if (j != gtid && j < N) {
                    float dx = myBody.x - shBodies[i].x;
                    float dy = myBody.y - shBodies[i].y;
                    float dist_sq = dx * dx + dy * dy;
                    float dist_cb = dist_sq * sqrtf(dist_sq);
                    float f_x = myBody.mass * shBodies[i].mass * dx / dist_cb;
                    float f_y = myBody.mass * shBodies[i].mass * dy / dist_cb;
                    myForce.x += f_x;
                    myForce.y += f_y;
                }
            }
            __syncthreads();
        }
        forces[gtid] = myForce;
    }
}

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

#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
    int N = argc > 1 ? stoi(argv[1]) : 1e5;
    vector<Body> bodies = generate_random_bodies(N);
    vector<Vector2D> forces(N);
    int print_total = 10;
    // Baseline
    cout << "Calculate gravitational forces between " << N << " random bodies:" << endl;
    if (falseg) {
        cout << "Brute force O(n^2) sequential approach:" << endl;
        auto start = std::chrono::high_resolution_clock::now();
        forces = brute_force_seq_n_body(bodies);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        cout << "Time taken: " << duration.count() / 1e6 << " s" << endl;
        for (int i = 0; i < N; i++) {
            if (print_total && (i + 1) % (N / print_total) == 0) {
                cout << "Body #" << i + 1 << " force: (" << forces[i].x << ", " << forces[i].y << ")" << endl;
            }
        }
        cout << endl;
    }
    // CUDA 
    cout << "Brute force CUDA parallel approach:" << endl;
    CUDABody *h_bodies = new CUDABody[N];
    CUDAVector2D *h_forces = new CUDAVector2D[N];
    for (int i = 0; i < N; i++) {
        h_bodies[i] = CUDABody(bodies[i].x, bodies[i].y, bodies[i].mass);
    }
    CUDABody *d_bodies;
    CUDAVector2D *d_forces;
    cudaMalloc(&d_bodies, N * sizeof(CUDABody));
    cudaMalloc(&d_forces, N * sizeof(CUDAVector2D));
    cudaMemcpy(d_bodies, h_bodies, N * sizeof(CUDABody), cudaMemcpyHostToDevice);
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    calculate_forces<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(CUDABody)>>>(d_bodies, d_forces, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_forces, d_forces, N * sizeof(CUDAVector2D), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        forces[i].x = (double) h_forces[i].x * grav;
        forces[i].y = (double) h_forces[i].y * grav;
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time taken: " << elapsedTime / 1000 << " s" << endl;
    for (int i = 0; i < N; i++) {
        if (print_total && (i + 1) % (N / print_total) == 0) {
            cout << "Body #" << i + 1 << " force: (" << forces[i].x << ", " << forces[i].y << ")" << endl;
        }
    }
    delete[] h_bodies;
    delete[] h_forces;
    cudaFree(d_bodies);
    cudaFree(d_forces);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}