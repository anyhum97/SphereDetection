#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Geometry>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Reflection.cu"
#include "ply.h"

#define LOG_ENABLE

using namespace ply;

unsigned int vertex = 0;

struct sphere
{
	ply::float3 center;

	float radius = 0.0f;
	float trust = 0.0f;
};

Reflection<ply::float3> Points;

Reflection<unsigned int> temp;

cudaEvent_t start;
cudaEvent_t stop;

__global__ void Detect(ply::float3* Points, unsigned int* temp, unsigned int vertex)
{
	// <<<Blocks, 256>>>

	const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;
	
	const unsigned int index = block*256 + thread;

    if(thread >= 256 || index >= vertex)
    {
        return;
    }

	temp[index] = index;	
}

void CudaMalloc(vector3f points)
{	
	vertex = points.size();

	if(vertex == 0)
	{
		return;
	}
	
	cudaSetDevice(0);

	Points = Malloc<ply::float3>(points.data(), vertex, true);

	temp = Malloc<unsigned int>(vertex);

	unsigned int size = 0;

	size += Points.size;

	std::cout << "Allocated " << size/(1024.0f*1024.0f) << "MB of GPU memory\n\n";
}

void CudaFree()
{
	Free(Points);

    cudaDeviceReset();
}

void Test()
{
	if(vertex == 0)
	{
		return;
	}

	////////////////////////////////////////////////////////////////////////

	const unsigned int threads = 256;

	////////////////////////////////////////////////////////////////////////

    cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

    ////////////////////////////////////////////////////////////////////////

	unsigned int block = static_cast<unsigned int>((float)vertex/threads+0.5f);

	while(block*threads < vertex)
	{
		++block;
	}

    Detect<<<block, threads>>>(Device(Points), Device(temp), vertex);

	Receive(temp);

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
	vector3f points;
	LoadVertex(&points, "Test_Sphere_Detector.ply");

	////////////////////////////////////////////////////////////////////////

    CudaMalloc(points);

    ////////////////////////////////////////////////////////////////////////

	Test();

    ////////////////////////////////////////////////////////////////////////

    CudaFree();

	system("pause");
	return 0;
}