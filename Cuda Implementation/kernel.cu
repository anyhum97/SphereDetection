#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Geometry>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Reflection.cu"

cudaEvent_t start;
cudaEvent_t stop;

void CudaMalloc()
{
    cudaSetDevice(0);
}

void CudaFree()
{
    cudaDeviceReset();
}

void Test()
{
    cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

    ////////////////////////////////////////////////////////////////////////

    

    ////////////////////////////////////////////////////////////////////////

    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time = 0;

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << time << "ms [OK]\n\n";
}

void main()
{
    CudaMalloc();

    ////////////////////////////////////////////////////////////////////////

	

    ////////////////////////////////////////////////////////////////////////

    CudaFree();
}