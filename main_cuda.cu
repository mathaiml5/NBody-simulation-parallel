#include <iostream>
#include <vector>
#include <cmath>
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
__device__ CUDAVector2D bodyBodyInteraction(CUDABody bi, CUDABody bj, CUDAVector2D fi) {
    float dx = bj.x - bi.x;
    float dy = bj.y - bi.y;
    float dist_sq = dx * dx + dy * dy;
    float dist_cb = dist_sq * sqrtf(dist_sq);
    float f_x = bi.mass * bj.mass * dx / dist_cb;
    float f_y = bi.mass * bj.mass * dy / dist_cb;
    fi.x += f_x;
    fi.y += f_y;
    return fi;
}

__device__ CUDAVector2D tile_calculation(CUDABody myBody, CUDAVector2D force) {
    extern __shared__ CUDABody shBodies[];
    #pragma unroll 4
    for (int i = 0; i < blockDim.x; i++) {
        if (myBody.x != shBodies[i].x || myBody.y != shBodies[i].y) {
            force = bodyBodyInteraction(myBody, shBodies[i], force);
        }
    }
    return force;
}

__global__ void calculate_forces(CUDABody *bodies, CUDAVector2D *forces, int N) {
    extern __shared__ CUDABody shBodies[];
    CUDABody myBody;
    CUDAVector2D myForce;
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
            myForce = tile_calculation(myBody, myForce);
            __syncthreads();
        }
        forces[gtid] = myForce;
    }
}

#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
    // CUDA 
    int N = argc > 1 ? stoi(argv[1]) : 1e5;
    vector<Body> bodies = generate_random_bodies(N);
    int print_total = 3;
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
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time taken: " << elapsedTime << " s" << endl;
    for (int i = 0; i < N; i++) {
        if (print_total && (i + 1) % (N / print_total) == 0) {
            cout << "Body #" << i + 1 << " force: (" << h_forces[i].x << ", " << h_forces[i].y << ")" << endl;
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