#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef std::vector<float3> vector3f;

#include "Reflection.cu"
#include "ply.h"

#define __local__ __host__ __device__

const unsigned int threadsPerBlock = 128;

unsigned int vertex = 0;

struct sphere
{
	float3 center;

	float radius;
	float trust;

	__local__ sphere()
	{
		center.x = 0.0f;
		center.y = 0.0f;
		center.z = 0.0f;

		radius = 0.0f;
		trust = 0.0f;
	}

	__local__ sphere(float3 center, float radius, float trust)
	{
		this->center = center;

		this->radius = radius;
		this->trust = trust;
	}
};

Reflection<float3> Points;

Reflection<sphere> Spheres1;
Reflection<sphere> Spheres2;

cudaEvent_t start;
cudaEvent_t stop;

////////////////////////////////////////////////////////////////////////

__device__ float det3(const float a11, const float a12, const float a13,
					  const float a21, const float a22, const float a23,
					  const float a31, const float a32, const float a33)
{
	return a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a31*a22*a13 - a32*a23*a11 - a33*a21*a12;
}

__device__ float det4(const float a11, const float a12, const float a13, const float a14,
		              const float a21, const float a22, const float a23, const float a24,
		              const float a31, const float a32, const float a33, const float a34,
		              const float a41, const float a42, const float a43, const float a44)
{
	return a11*det3(a22, a23, a24, a32, a33, a34, a42, a43, a44) -
		   a21*det3(a12, a13, a14, a32, a33, a34, a42, a43, a44) +
		   a31*det3(a12, a13, a14, a22, a23, a24, a42, a43, a44) - 
		   a41*det3(a12, a13, a14, a22, a23, a24, a32, a33, a34);
}

__device__ float Sphere(float3 points[4], float3& center, float& radius, const float eps = 1e-3f)
{
	/* This function calculates with what reliability 4 points belong to the sphere.
	* If this value exceeds the threshold, then the coordinate of the center of the 
	* sphere and its radius are calculated.
	* For more information check https://mathworld.wolfram.com/Circumsphere.html
	*/

	const float x1 = points[0].x;
	const float y1 = points[0].y;
	const float z1 = points[0].z;

	const float x2 = points[1].x;
	const float y2 = points[1].y;
	const float z2 = points[1].z;

	const float x3 = points[2].x;
	const float y3 = points[2].y;
	const float z3 = points[2].z;

	const float x4 = points[3].x;
	const float y4 = points[3].y;
	const float z4 = points[3].z;

	const float detA = det4(x1, y1, z1, 1.0f,
							x2, y2, z2, 1.0f,
							x3, y3, z3, 1.0f,
							x4, y4, z4, 1.0f);

	const float trust = abs(detA);

	if(trust < eps)
	{
		// This value does not exceed the threshold.
		// This means that the points do not lie on the sphere.

		return 0.0f;
	}

	const float a1 = x1*x1 + y1*y1 + z1*z1;
	const float a2 = x2*x2 + y2*y2 + z2*z2;
	const float a3 = x3*x3 + y3*y3 + z3*z3;
	const float a4 = x4*x4 + y4*y4 + z4*z4;

	const float detX = det4(a1, y1, z1, 1.0f,
							a2, y2, z2, 1.0f,
							a3, y3, z3, 1.0f,
							a4, y4, z4, 1.0f);

	const float detY = -det4(a1, x1, z1, 1.0f,
							 a2, x2, z2, 1.0f,
							 a3, x3, z3, 1.0f,
							 a4, x4, z4, 1.0f);

	const float detZ = det4(a1, x1, y1, 1.0f,
							a2, x2, y2, 1.0f,
							a3, x3, y3, 1.0f,
							a4, x4, y4, 1.0f);

	const float detC = det4(a1, x1, y1, z1,
							a2, x2, y2, z2,
							a3, x3, y3, z3,
							a4, x4, y4, z4);

	center.x = 0.5f*detX/detA;
	center.y = 0.5f*detY/detA;
	center.z = 0.5f*detZ/detA;
	
	radius = 0.5f*sqrt(detX*detX + detY*detY + detZ*detZ - 4.0f*detA*detC)/trust;

	return trust;
}

__global__ void Detect(float3* Points, sphere* Spheres1, sphere* Spheres2, const unsigned int vertex)
{
	// <<<Blocks, threads>>>

	const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;
	
	const unsigned int index = block*threadsPerBlock + thread;

    if(thread >= threadsPerBlock || index >= vertex)
    {
        return;
    }

	const float ax = Points[index].x;
	const float ay = Points[index].y;
	const float az = Points[index].z;

	float range1 = FLT_MAX;
	float range2 = FLT_MAX;

	float3 p1[24];
	float3 p2[24];

	for(unsigned int j=0; j<vertex; ++j)
	{
		const float bx = Points[j].x;
		const float by = Points[j].y;
		const float bz = Points[j].z;

		const float distance = sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz));

		if(bz > az)
		{
			if(distance < range1)
			{
				range1 = distance;
			}
		}

		if(bz < az)
		{
			if(distance < range2)
			{
				range2 = distance;
			}
		}
	}

	unsigned int count1 = 0;
	unsigned int count2 = 0;

	for(unsigned int j=0; j<vertex; ++j)
	{
		if(j != index)
		{
			const float bx = Points[j].x;
			const float by = Points[j].y;
			const float bz = Points[j].z;

			const float distance = sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz));

			if(distance > 2.0f && distance < 10.0f)
			{
				if(distance < range1)
				{
					if(count1 < 24)
					{
						p1[count1] = Points[j];
						++count1;
					}
				}

				if(distance < range2)
				{
					if(count2 < 24)
					{
						p2[count2] = Points[j];
						++count2;
					}
				}
			}
		}
	}

	float3 p4[4];

	if(count1 >= 6)
	{
		sphere local[8];

		const unsigned int attempts1 = count1 / 3;

		unsigned int index = 0;

		for(unsigned int j=0; j<attempts1 && j<8; ++j)
		{
			p4[0] = Points[index];

			p4[1] = p1[3*j+0];
			p4[2] = p1[3*j+1];
			p4[3] = p1[3*j+2];

			float3 center;
			float radius = 0.0f;

			float trust = Sphere(p4, center, radius, 5.0f);

			if(trust > 5.0f && radius > 10.0f && radius < 60.0f)
			{
				local[index] = sphere(center, radius, trust);
				++index;
			}				
		}

		if(index)
		{
			sphere best;

			float factor = 0.0f;

			for(unsigned int j=0; j<index; ++j)
			{
				if(local[j].trust > factor)
				{
					factor = local[j].trust;
					best = local[j];
				}
			}

			Spheres1[index] = best;
		}
	}

	if(count2 >= 6)
	{
		sphere local[8];

		const unsigned int attempts2 = count2 / 3;

		unsigned int index = 0;

		for(unsigned int j=0; j<attempts2 && j<8; ++j)
		{
			p4[0] = Points[index];

			p4[1] = p1[3*j+0];
			p4[2] = p1[3*j+1];
			p4[3] = p1[3*j+2];

			float3 center;
			float radius = 0.0f;

			float trust = Sphere(p4, center, radius, 5.0f);

			if(trust > 5.0f && radius > 10.0f && radius < 60.0f)
			{
				local[index] = sphere(center, radius, trust);
				++index;
			}				
		}

		if(index)
		{
			sphere best;

			float factor = 0.0f;

			for(unsigned int j=0; j<index; ++j)
			{
				if(local[j].trust > factor)
				{
					factor = local[j].trust;
					best = local[j];
				}
			}

			Spheres2[index] = best;
		}
	}
}

////////////////////////////////////////////////////////////////////////

bool TrustCompare(sphere sphere1, sphere sphere2)
{
	// This function compares sphere trust factor for sorting.

	return sphere1.trust > sphere2.trust;
}

////////////////////////////////////////////////////////////////////////

bool LoadVertex(vector3f* points, std::string path)
{
	std::ifstream text(path, std::ios::binary);

	char header[2048];

	text.read(header, 2048);

	std::string str(header);
	
	int index1 = str.find("ply");
	int index2 = str.find("PLY");
	
	if(index1 < 0 && index2 < 0)
	{
		return false;
	}

	int index3 = str.find("vertex")+strlen("vertex");
	int index4 = str.find('\n', index3);
	int index5 = str.find("end_header\n")+strlen("end_header\n");

	if(index4-index3 < 3 || index5 < 0)
	{
		return false;
	}

	std::string str2 = str.substr(index3, index4-index3);

	int vertex = std::stoi(str2);

	int size = vertex*sizeof(float3);

	text.close();

	std::ifstream bin(path, std::ios::binary);

	bin.seekg(index5);

	float3* buf = new float3[vertex];

	points->clear();

	bin.read((char*)buf, size);

	points->assign(buf, buf+vertex);

	delete []buf;

	bin.close();

	return points->size();
}

////////////////////////////////////////////////////////////////////////

void CudaMalloc(vector3f points)
{	
	vertex = points.size();

	if(vertex == 0)
	{
		return;
	}
	
	cudaSetDevice(0);

	Points = Malloc<float3>(points.data(), vertex, true);
	
	Spheres1 = Malloc<sphere>(vertex);
	Spheres2 = Malloc<sphere>(vertex);
	
	unsigned int size = 0;

	size += Points.size;

	size += Spheres1.size;
	size += Spheres2.size;

	std::cout << "Allocated " << size/(1024.0f*1024.0f) << "MB of GPU memory\n\n";
}

void CudaFree()
{
	Free(Points);

	Free(Spheres1);
	Free(Spheres2);

    cudaDeviceReset();
}

void SphereDetectionAlgorithm(unsigned int nSpheres)
{
	////////////////////////////////////////////////////////////////////////

	unsigned int block = static_cast<unsigned int>((float)vertex/threadsPerBlock+0.5f);

	while(block*threadsPerBlock < vertex)
	{
		++block;
	}

    Detect<<<block, threadsPerBlock>>>(Device(Points), Device(Spheres1), Device(Spheres2), vertex);
	
	Receive(Spheres1);
	Receive(Spheres2);

	////////////////////////////////////////////////////////////////////////

	std::vector<sphere> spheres1;
	std::vector<sphere> spheres2;

	for(int i=0; i<vertex; ++i)
	{
		if(Spheres1.host[i].trust > 0.0f)
		{
			spheres1.push_back(Spheres1.host[i]);
		}

		if(Spheres2.host[i].trust > 0.0f)
		{
			spheres2.push_back(Spheres2.host[i]);
		}
	}

	std::sort(spheres1.begin(), spheres1.end(), TrustCompare);
	std::sort(spheres2.begin(), spheres2.end(), TrustCompare);

	std::ofstream file1("detected1.txt");

	for(int i=0; i<64 && i<spheres1.size(); ++i)
	{
		file1 << "(";

		file1 << spheres1[i].center.x << ", ";
		file1 << spheres1[i].center.y << ", ";
		file1 << spheres1[i].center.z << "): ";

		file1 << spheres1[i].radius << "\t";
		file1 << spheres1[i].trust << "\n\n";
	}

	file1.close();

	std::ofstream file2("detected2.txt");

	for(int i=0; i<64 && i<spheres2.size(); ++i)
	{
		file2 << "(";

		file2 << spheres2[i].center.x << ", ";
		file2 << spheres2[i].center.y << ", ";
		file2 << spheres2[i].center.z << "): ";

		file2 << spheres2[i].radius << "\t";
		file2 << spheres2[i].trust << "\n\n";
	}

	file2.close();

	std::ofstream file3("solution1.txt");

	for(int i=0; i<nSpheres && i<spheres1.size(); ++i)
	{
		file3 << "(";

		file3 << spheres1[i].center.x << ", ";
		file3 << spheres1[i].center.y << ", ";
		file3 << spheres1[i].center.z << "): ";

		file3 << spheres1[i].radius << "\n\n";
	}

	file3.close();

	std::ofstream file4("solution2.txt");

	for(int i=0; i<nSpheres && i<spheres2.size(); ++i)
	{
		file4 << "(";

		file4 << spheres1[i].center.x << ", ";
		file4 << spheres1[i].center.y << ", ";
		file4 << spheres1[i].center.z << "): ";

		file4 << spheres1[i].radius << "\n\n";
	}

	file4.close();
}

void Test()
{
	if(vertex == 0)
	{
		return;
	}

	////////////////////////////////////////////////////////////////////////

    cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

    ////////////////////////////////////////////////////////////////////////

	SphereDetectionAlgorithm(6);

    ////////////////////////////////////////////////////////////////////////

    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time = 0;

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << time << " ms [OK]\n\n";
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

	//system("pause");
	return 0;
}