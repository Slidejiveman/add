// reference: https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

// __global__ specifies a kernel in CUDA. It specifies that
// this function runs on the GPU but can be called from the
// CPU. Code that runs on the GPU is called device code.
// code that runs on the CPU is called host code.
__global__ void add(int n, float* x, float* y) {
	
	// Adding an index and a stride is how you tell
	// the kernel not to run through the entiretly of
	// the array. Starting at index, each thread will
	// run a certain number of iterations that are
	// stride apart. 
	// The calculation for the index and stride are
	// idiomatic CUDA. They are finding the thread
	// via its offset from the start of all the threads
	// Note: this type of loop is called a grid-stride loop
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}
}

int main()
{
	int N = 1 << 20; // 1M elements
	float* x;//= new float[N]; This is how we do it for the CPU
	float* y;//= new float[N];

			 // To make these variables in Unified Memory, which is
			 // memory that can be accessed by all CPUs and GPUs on
			 // the system, we use cudaMallocManaged
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize arrays
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// A function the GPU can run is called a kernel in CUDA
	// This function is the kernel in this example
	// triple angle brackets launches the add kernel on the 
	// GPU. Changing the values in here configures the
	// number of usable threads. The number on the right
	// can be increased by multiples of 32 because that is
	// what CUDA GPUs do. The second parameter is the number
	// of threads. The first parameter is the number of
	// thread blocks. There are 256 threads available per 
	// block.
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add<<<numBlocks, blockSize>>>(N, x, y);

	// wait for GPU to finish before accessing on CPU
	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f(
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	// Free memory
	//delete[] x; This is how we do it on a CPU
	//delete[] y;

	// cudaFree frees resources in Unified Memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}


