C = g++
CFLAGS = -std=c++17 -Wall -O2 -fopenmp
OBJ_FILES = utils.o methods.o main.o

NVCC = nvcc
NVCCFLAGS = -std=c++14 -O2
CUDA_INCLUDE = -I/usr/local/cuda-11.1/include
CUDA_LIBS = -L/usr/local/cuda-11.1/lib64 -lcudart_static -lrt -lpthread -ldl
CUDA_ARCH = -gencode=arch=compute_86,code=sm_86

all: nbody

nbody: $(OBJ_FILES)
	$(C) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(C) $(CFLAGS) -c $< -o $@

nbody_cuda: main_cuda.o utils.o 
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(CUDA_LIBS) -Xcompiler -fPIC

main_cuda.o: main_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CUDA_INCLUDE) -dc $< -o $@

clean:
	rm -f nbody nbody_cuda $(OBJ_FILES)