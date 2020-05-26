#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <Eigen/Geometry>

unsigned int vertex = 1024;

struct float3
{
	float x;
	float y;
	float z;
};

std::vector<float3> points;		//	[vertex];

void UpdateVertexCount(unsigned int& vertex, unsigned int count)
{
	// This function determines how many vertex-points to use.
	
	// Monte-Carlo optimization:
	// We will use only ~5% of loaded points.

	// 1) vertex - how many points to use.
	// This value must be a power of two to more efficiently 
	// parallelize tasks on the CUDA version.

	// 2) count is a total number of loaded points.
	// This value cannot be less than 512.

	if(count < vertex)
	{
		vertex = 512;
	}
	else
	{
		if(vertex*20 < count)
		{
			vertex = count / 20;

			// Alignment to the power of two:

			unsigned int power = 1;

			while(vertex > power)
			{
				power <<= 1;
			}

			vertex = power;
		}
	}
}

void LoadFile(std::vector<float3>* selected_points)
{
	// This function loads vertex-points and select points to use.

	std::ifstream file1("points.txt");

	if(!file1)
	{
		vertex = 0;
		return;
	}

	unsigned int count = 0;

	while(file1)
	{
		float3 point;

		file1 >> point.x >> point.y >> point.z;

		points.push_back(point);

		++count;
	}

	file1.close();
	
	// Request at least 512 points:

	if(count < 512)	
	{	
		count = 0;
		return;
	}

	UpdateVertexCount(vertex, count);

	std::ofstream file2("samples.txt");

	// Evenly copy vertex-points:

	for(int i=0; i<vertex; ++i)
	{
		float3 point = points[rand()%count];

		selected_points->push_back(point);

		// We can save selected points in text format to explore them:

		file2 << point.x << "\t";
		file2 << point.y << "\t";
		file2 << point.z << "\n";
	}

	file2.close();
}

int main()
{
	std::vector<float3> selected_points;

	LoadFile(&selected_points);

	//unsigned int vertex = 768;
	//unsigned int count = 132000;

	//UpdateVertexCount(vertex, count);

	return 0;
}