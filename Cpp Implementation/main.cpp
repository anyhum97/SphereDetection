#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include <Eigen/Geometry>

//#define LOG_ENABLE

unsigned int vertex = 1024;	// Number of points to use. Represents the power of two.

const float DUTY_CYCLE = 0.05f;	// Percentage of points to use [0..1];

float size1 = 0.0f;
float size2 = 0.0f;
float size3 = 0.0f;

float min1 = 0.0f;
float max1 = 0.0f;

float min2 = 0.0f;
float max2 = 0.0f;

float min3 = 0.0f;
float max3 = 0.0f;

struct float3
{
	float x;
	float y;
	float z;
};

struct sphere
{
	float3 center;

	float radius;
	float trust;
};

typedef std::vector<float3> vector3f;

float len(float3 point)
{
	return sqrt((point.x*point.x) + (point.y*point.y) + (point.z*point.z));
}

float Sphere(float3 points[4], float3& center, float& radius, const float eps = 1e-3f)
{
	// This function determines whether the given 4 points lie on the sphere.
	// If yes, then calculates the center and radius of the sphere.
	// For more information check https://mathworld.wolfram.com/Circumsphere.html

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

	Eigen::Matrix4f a;

	a(0, 0) = x1;
	a(1, 0) = x2;
	a(2, 0) = x3;
	a(3, 0) = x4;

	a(0, 1) = y1;
	a(1, 1) = y2;
	a(2, 1) = y3;
	a(3, 1) = y4;

	a(0, 2) = z1;
	a(1, 2) = z2;
	a(2, 2) = z3;
	a(3, 2) = z4;

	a(0, 3) = 1.0f;
	a(1, 3) = 1.0f;
	a(2, 3) = 1.0f;
	a(3, 3) = 1.0f;

	const float detA = a.determinant();

	if(abs(detA) < eps)
	{
		return 0.0f;
	}

	const float a1 = x1*x1 + y1*y1 + z1*z1;
	const float a2 = x2*x2 + y2*y2 + z2*z2;
	const float a3 = x3*x3 + y3*y3 + z3*z3;
	const float a4 = x4*x4 + y4*y4 + z4*z4;

	Eigen::Matrix4f DetX;
	Eigen::Matrix4f DetY;
	Eigen::Matrix4f DetZ;

	Eigen::Matrix4f c;

	DetX(0, 0) = a1;
	DetX(1, 0) = a2;
	DetX(2, 0) = a3;
	DetX(3, 0) = a4;

	DetX(0, 1) = y1;
	DetX(1, 1) = y2;
	DetX(2, 1) = y3;
	DetX(3, 1) = y4;

	DetX(0, 2) = z1;
	DetX(1, 2) = z2;
	DetX(2, 2) = z3;
	DetX(3, 2) = z4;

	DetX(0, 3) = 1.0f;
	DetX(1, 3) = 1.0f;
	DetX(2, 3) = 1.0f;
	DetX(3, 3) = 1.0f;

	DetY(0, 0) = a1;
	DetY(1, 0) = a2;
	DetY(2, 0) = a3;
	DetY(3, 0) = a4;

	DetY(0, 1) = x1;
	DetY(1, 1) = x2;
	DetY(2, 1) = x3;
	DetY(3, 1) = x4;

	DetY(0, 2) = z1;
	DetY(1, 2) = z2;
	DetY(2, 2) = z3;
	DetY(3, 2) = z4;

	DetY(0, 3) = 1.0f;
	DetY(1, 3) = 1.0f;
	DetY(2, 3) = 1.0f;
	DetY(3, 3) = 1.0f;

	DetZ(0, 0) = a1;
	DetZ(1, 0) = a2;
	DetZ(2, 0) = a3;
	DetZ(3, 0) = a4;

	DetZ(0, 1) = x1;
	DetZ(1, 1) = x2;
	DetZ(2, 1) = x3;
	DetZ(3, 1) = x4;

	DetZ(0, 2) = y1;
	DetZ(1, 2) = y2;
	DetZ(2, 2) = y3;
	DetZ(3, 2) = y4;

	DetZ(0, 3) = 1.0f;
	DetZ(1, 3) = 1.0f;
	DetZ(2, 3) = 1.0f;
	DetZ(3, 3) = 1.0f;

	c(0, 0) = a1;
	c(1, 0) = a2;
	c(2, 0) = a3;
	c(3, 0) = a4;

	c(0, 1) = x1;
	c(1, 1) = x2;
	c(2, 1) = x3;
	c(3, 1) = x4;

	c(0, 2) = y1;
	c(1, 2) = y2;
	c(2, 2) = y3;
	c(3, 2) = y4;

	c(0, 3) = z1;
	c(1, 3) = z2;
	c(2, 3) = z3;
	c(3, 3) = z4;

	const float detX = DetX.determinant();
	const float detY = -DetY.determinant();
	const float detZ = DetZ.determinant();

	const float detC = c.determinant();

	center.x = 0.5f*detX/detA;
	center.y = 0.5f*detY/detA;
	center.z = 0.5f*detZ/detA;

	radius = 0.5f*sqrt(detX*detX + detY*detY + detZ*detZ - 4.0f*detA*detC)/abs(detA);

	return abs(detA);
}

void UpdateVertexCount(unsigned int& vertex, unsigned int count)
{
	// This function determines how many vertex-points to use.
	
	// Monte-Carlo optimization:
	// We will use only DUTY_CYCLE of loaded points.

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
		if(vertex < DUTY_CYCLE*count)
		{
			vertex = DUTY_CYCLE*count;

			// Alignment to the power of 2:

			unsigned int power = 1;

			while(vertex > power)
			{
				power <<= 1;
			}

			vertex = power;
		}
	}
}

void LoadFile(vector3f* selected_points)
{
	// This function loads vertex-points and select points to use.

	vector3f points;

	std::ifstream file1("points.txt");

	if(!file1)
	{
		vertex = 0;
		return;
	}

	unsigned int count = 0;

	// Determine the linear dimensions for greater reliability

	min1 = FLT_MAX;
	max1 = -FLT_MAX;

	min2 = FLT_MAX;
	max2 = -FLT_MAX;

	min3 = FLT_MAX;
	max3 = -FLT_MAX;

	while(file1)
	{
		float3 point;

		file1 >> point.x >> point.y >> point.z;

		points.push_back(point);

		min1 = point.x < min1 ? point.x : min1;
		min2 = point.y < min2 ? point.y : min2;
		min3 = point.z < min3 ? point.z : min3;

		max1 = point.x > max1 ? point.x : max1;
		max2 = point.y > max2 ? point.y : max2;
		max3 = point.z > max3 ? point.z : max3;

		++count;
	}

	file1.close();
	
	size1 = abs(max1-min1);
	size2 = abs(max2-min2);
	size3 = abs(max3-min3);

	// Request at least 512 points:

	if(count < 512)	
	{	
		count = 0;
		return;
	}

	UpdateVertexCount(vertex, count);

	srand(0);

#if defined(LOG_ENABLE)

	std::ofstream file2("samples.txt");

	// Evenly copy vertex-points:

	for(int i=0; i<vertex; ++i)
	{
		float3 point = points[rand()%count];

		selected_points->push_back(point);

		// We can save selected points in text format to explore:

		file2 << point.x << "\t";
		file2 << point.y << "\t";
		file2 << point.z << "\n";
	}

	file2.close();

#else

	// Evenly copy vertex-points:

	for(int i=0; i<vertex; ++i)
	{
		selected_points->push_back(points[rand()%count]);
	}

#endif
}

bool Compare(sphere sphere1, sphere sphere2)
{
	const float value1 = len(sphere1.center) + sphere1.radius;
	const float value2 = len(sphere2.center) + sphere2.radius;

	return value1 < value2;
}

bool TrustCompare(sphere sphere1, sphere sphere2)
{
	return sphere1.trust > sphere2.trust;
}

bool near(const float a, const float b, const float e = 1e-3f)
{
	if(a + e >= b && a - e <= b)
	{
		return true;
	}

	return false;
}

bool Equals(sphere sphere1, sphere sphere2)
{
	//return (near(sphere1.center.x, sphere2.center.x) &&
	//		near(sphere1.center.y, sphere2.center.y) &&
	//		near(sphere1.center.z, sphere2.center.z) &&
	//		near(sphere1.radius, sphere2.radius));

	return (sphere1.center.x == sphere2.center.x) &&
		   (sphere1.center.y == sphere2.center.y) &&
		   (sphere1.center.z == sphere2.center.z) &&
		   (sphere1.radius == sphere2.radius);
}

void Detect(vector3f points, unsigned int amount)
{
	// This function use vertex-points to detect spheres.

	if(vertex < 512 || points.size() != vertex)
	{
		return;
	}

	// Find the distance between all points:

	std::vector<std::vector<float>> distance;
	
	float mcd = 0.0f; // Middle closest distance.

	float counter = 0.0f;

	for(int i=0; i<vertex; ++i)
	{
		std::vector<float> buf;

		float min = FLT_MAX;

		for(int j=0; j<vertex; ++j)
		{
			if(i != j)
			{
				const float distance = sqrt((points[i].x - points[j].x)*(points[i].x - points[j].x) + 
											(points[i].y - points[j].y)*(points[i].y - points[j].y) + 
											(points[i].z - points[j].z)*(points[i].z - points[j].z));

				buf.push_back(distance);

				if(distance < min)
				{
					min = distance;
				}
			}
			else
			{
				buf.push_back(FLT_MAX);
			}		
		}
		
		distance.push_back(buf);

		mcd = (mcd*counter + min)/(counter+1.0f);

		counter += 1.0f;
	}

	// Looking for points that are local extrema:

	bool* IsMax = new bool[vertex];
	bool* IsMin = new bool[vertex];

	const float factor = 10.0f*mcd;

	for(int i=0; i<vertex; ++i)
	{
		const float z1 = points[i].z;

		IsMax[i] = true;
		IsMin[i] = true;

		for(int j=0; j<vertex; ++j)
		{
			const float z2 = points[j].z;

			if(distance[i][j] < factor)
			{
				if(z1 < z2)
				{
					IsMax[i] = false;
				}

				if(z1 > z2)
				{
					IsMin[i] = false;
				}
			}

			if(IsMax[i] == false && IsMin[i] == false)
			{
				break;
			}
		}
	}

	std::vector<unsigned int> peaks1;
	std::vector<unsigned int> peaks2;

#if defined(LOG_ENABLE)

	std::ofstream file1("max.txt");
	std::ofstream file2("min.txt");

	for(int i=0; i<vertex; ++i)
	{
		if(IsMax[i])
		{
			peaks1.push_back(i);
			
			file1 << points[i].x << ", " << points[i].y << ", " << points[i].z << "\n";
		}

		if(IsMin[i])
		{
			peaks2.push_back(i);

			file2 << points[i].x << ", " << points[i].y << ", " << points[i].z << "\n";
		}
	}
	
	file1.close();
	file2.close();

#else

	for(int i=0; i<vertex; ++i)
	{
		if(IsMax[i])
		{
			peaks1.push_back(i);
		}

		if(IsMin[i])
		{
			peaks2.push_back(i);
		}
	}

#endif

	delete[] IsMax;
	delete[] IsMin;

	// Ñheck if the found extrema are points of the sphere
	// For this select three more points in the neighborhood and 
	// ñall the function Sphere();

	// Write to a file found center and radius of spheres.

	std::vector<sphere> spheres;

	for(int i=0; i<peaks1.size(); ++i)
	{
		float3 p4[4];

		unsigned int index = peaks1[i];

		p4[0] = points[index];

		unsigned int count = 1;

		float r2 = 5.0f*mcd;

		while(count != 4)
		{
			for(int j=0; j<vertex && count < 4; ++j)
			{
				if(distance[index][j] < r2 && distance[index][j] > 1.0f)
				{
					p4[count] = points[j];
					++count;
				}
			}

			r2 += 1.0f;
		}

		float3 center;
		float radius = 0.0f;

		float trust = Sphere(p4, center, radius, 0.5f);
		
		sphere sphere1;

		sphere1.center = center;
		sphere1.radius = radius;
		sphere1.trust = trust;

		float size = std::max(size1, std::max(size2, size3));

		if(radius > 2.0f*mcd && radius < 50.0f && trust > 0.5f)
		{
			spheres.push_back(sphere1);
		}
	}

	std::sort(spheres.begin(), spheres.end(), Compare);

	spheres.erase(std::unique(spheres.begin(), spheres.end(), Equals), spheres.end());

	std::ofstream file3("unsorted.txt");

	for(int i=0; i<spheres.size(); ++i)
	{
		file3 << "(";

		file3 << spheres[i].center.x << ", ";
		file3 << spheres[i].center.y << ", ";
		file3 << spheres[i].center.z << "): ";

		file3 << spheres[i].radius << "\n\n";
	}

	file3.close();

	std::sort(spheres.begin(), spheres.end(), TrustCompare);

	std::ofstream file4("sorted.txt");

	for(int i=0; i<amount && i<spheres.size(); ++i)
	{
		file4 << "(";

		file4 << spheres[i].center.x << ", ";
		file4 << spheres[i].center.y << ", ";
		file4 << spheres[i].center.z << "): ";

		file4 << spheres[i].radius << "\n\n";
	}

	file4.close();
}

int main()
{
	vector3f points;

	LoadFile(&points);
	Detect(points, 6);

	return 0;
}