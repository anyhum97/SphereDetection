#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Geometry>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Reflection.h"
#include "ply.h"

using namespace ply;

unsigned int vertex = 1024;

struct sphere
{
	ply::float3 center;

	float radius = 0.0f;
	float trust = 0.0f;
};

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

int main()
{
    CudaMalloc();

    ////////////////////////////////////////////////////////////////////////

	

    ////////////////////////////////////////////////////////////////////////

    CudaFree();
	return 0;
}