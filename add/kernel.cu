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

// Moves the initialization of the arrays to the GPU.
// This reduces page fault issues when performing the
// add kernel
__global__ void init(int n, float* x, float* y) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		x[i] = 1.0f;
		y[i] = 2.0f;
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

	// The following code is what I would use if I was prefetching:
	/*
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(x, N * sizeof(float), device, NULL);
	cudaMemPrefetchAsync(y, N * sizeof(float), device, NULL);
	*/

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

	// Create events to allow performance measuring
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// initialize the arrays. I've decided to do this on the GPU
	// rather than prefetch. Either way, I would have prevented 
	// losing performance to page faults.
	init << <numBlocks, blockSize >> > (N, x, y);

	// time the main calculation
	cudaEventRecord(start);
	add<<<numBlocks, blockSize>>>(N, x, y);
	cudaEventRecord(stop);

	// wait for GPU to finish before accessing on CPU. With old GPUs,
	// forgetting this step would likely result in a segmentation fault.
	// If using a Pascal-type GPU, page faults are supported, so the CPUs,
	// and GPUs can simultaneously access the memory. You still need this
	// call on a Pascal GPU to avoid reading invalid data (race condition)
	// A call to this function is also necessary to measure kernel execution
	// time as opposed to kernel launch time.
	//cudaDeviceSynchronize();

	// This method blocks execution until an event
	// is recorded!
	cudaEventSynchronize(stop);

	// Collect the elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Check for errors (all values should be 3.0f(
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl
		<< "Elapsed Time: " << milliseconds << std::endl;

	// Free memory
	//delete[] x; This is how we do it on a CPU
	//delete[] y;

	// cudaFree frees resources in Unified Memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}


