C = g++
CFLAGS = -std=c++17 -Wall -O2 -fopenmp
OBJ_FILES = utils.o methods.o main.o

NVCC = nvcc
NVCCFLAGS = -std=c++14 -O2
CUDA_ARCH = -arch=sm_60
CUDA_INCLUDE = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart

all: nbody

nbody: $(OBJ_FILES)
	$(C) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(C) $(CFLAGS) -c $< -o $@

nbody_cuda: main_cuda.o utils.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(CUDA_LIBS)

main_cuda.o: main_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CUDA_INCLUDE) -c $< -o $@

clean:
	rm -f nbody nbody_cuda $(OBJ_FILES)