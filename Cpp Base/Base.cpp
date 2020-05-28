#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "Eigen/Geometry"

#include "ply.h"

using namespace ply;

unsigned int vertex = 0;

struct sphere
{
	float3 center;

	float radius;
	float trust;

	sphere()
	{
		center.x = 0.0f;
		center.y = 0.0f;
		center.z = 0.0f;

		radius = 0.0f;
		trust = 0.0f;
	}

	sphere(float3 center, float radius, float trust)
	{
		this->center = center;

		this->radius = radius;
		this->trust = trust;
	}
};

float det3(const float a11, const float a12, const float a13,
		   const float a21, const float a22, const float a23,
		   const float a31, const float a32, const float a33)
{
	return (a11*a22*a33 + a12*a23*a31 + a13*a21*a32) - (a31*a22*a13 + a32*a23*a11 + a33*a21*a12);
}

float det4(const float a11, const float a12, const float a13, const float a14,
		   const float a21, const float a22, const float a23, const float a24,
		   const float a31, const float a32, const float a33, const float a34,
		   const float a41, const float a42, const float a43, const float a44)
{
	return a11*det3(a22, a23, a24, a32, a33, a34, a42, a43, a44) -
		   a21*det3(a12, a13, a14, a32, a33, a34, a42, a43, a44) +
		   a31*det3(a12, a13, a14, a22, a23, a24, a42, a43, a44) - 
		   a41*det3(a12, a13, a14, a22, a23, a24, a32, a33, a34);
}

float Sphere1(float3 points[4], float3& center, float& radius, const float eps = 1e-3f)
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

	// trust-factor is a probability that points lie on a sphere.
	// The larger it is, the better the points fit to the sphere.
	// trust-factor can be more than 1.0f.

	const float trust = abs(detA);

	if(trust < eps)
	{
		// trust-factor does not exceed the threshold.
		// This means that the points do not lie on the sphere.

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

	radius = 0.5f*sqrt(detX*detX + detY*detY + detZ*detZ - 4.0f*detA*detC)/trust;

	return trust;
}

float Sphere2(float3 points[4], float3& center, float& radius, const float eps = 1e-3f)
{
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

	// trust-factor is a probability that points lie on a sphere.
	// The larger it is, the better the points fit to the sphere.
	// trust-factor can be more than 1.0f.

	if(trust < eps)
	{
		// trust-factor does not exceed the threshold.
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

bool TrustCompare(sphere sphere1, sphere sphere2)
{
	// This function compares sphere trust factor for sorting.

	return sphere1.trust > sphere2.trust;
}

void LoadPlyFile(vector3f* points)
{
	//This function loads all vertex points.

	if(!LoadVertex(points, "Test_Sphere_Detector.ply"))
	{
		vertex = 0;
		return;
	}

	vertex = points->size();
}

void Detect(vector3f points, unsigned int nSpheres)
{
	// This function use all vertex-points to detect spheres.

	if(vertex < 512 || points.size() < vertex)
	{
		return;
	}

	std::vector<sphere> spheres;

	for(unsigned int i=0; i<vertex; ++i)
	{
		const float ax = points[i].x;
		const float ay = points[i].y;
		const float az = points[i].z;

		// Here we want to determine at what distance the point is a 
		// local max or min of projection to XY plane.
		// This is necessary to find abnormal peaks.

		// In other situations the axis might be different.
		// However we know the topology of the problem.

		// In any case, you need to take the projection onto the plane with maximum faces.

		float range1 = FLT_MAX;
		float range2 = FLT_MAX;

		for(unsigned int j=0; j<vertex; ++j)
		{
			const float bx = points[j].x;
			const float by = points[j].y;
			const float bz = points[j].z;

			const float distance = sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz));

			if(bz > az)
			{
				// Some point larger than ours means the area of local max is narrowing.

				if(distance < range1)
				{
					range1 = distance;
				}
			}

			if(bz < az)
			{
				// Some point less than ours means the area of local min is narrowing.

				if(distance < range2)
				{
					range2 = distance;
				}
			}
		}

		float3 p1[24];
		float3 p2[24];

		unsigned int count1 = 0;
		unsigned int count2 = 0;

		// Here we collect points that lie in the range of the local max and min.

		for(unsigned int j=0; j<vertex; ++j)
		{
			if(j != i)
			{
				const float bx = points[j].x;
				const float by = points[j].y;
				const float bz = points[j].z;

				const float distance = sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz));

				if(distance > 2.0f && distance < 10.0f)
				{
					if(distance < range1)
					{
						if(count1 < 24)
						{
							p1[count1] = points[j];
							++count1;
						}
					}

					if(distance < range2)
					{
						if(count2 < 24)
						{
							p2[count2] = points[j];
							++count2;
						}
					}
				}
			}
		}

		// Now we can check if the points lie on any sphere
		// and choose the most reliable case for this we will
		// compare trust-factors of each case:

		if(count1 >= 6)
		{
			sphere local[8];

			const unsigned int attempts1 = count1 / 3;

			unsigned int index = 0;

			for(unsigned int j=0; j<attempts1 && j<8; ++j)
			{
				float3 p4[4];

				p4[0] = points[i];

				p4[1] = p1[3*j+0];
				p4[2] = p1[3*j+1];
				p4[3] = p1[3*j+2];

				float3 center;
				float radius = 0.0f;

				float trust = Sphere2(p4, center, radius, 2.0f);

				if(trust > 2.0f && radius > 10.0f && radius < 60.0f)
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

				spheres.push_back(best);
			}			
		}
	}

	// Sort the found spheres by trust-factor
	std::sort(spheres.begin(), spheres.end(), TrustCompare);

	// Record all found spheres:

	std::ofstream file1("detected.txt");

	for(int i=0; i<64 && i<spheres.size(); ++i)
	{
		file1 << "(";

		file1 << spheres[i].center.x << ", ";
		file1 << spheres[i].center.y << ", ";
		file1 << spheres[i].center.z << "): ";

		file1 << spheres[i].radius << "\t";
		file1 << spheres[i].trust << "\n\n";
	}

	file1.close();

	// Record the best matches:

	std::ofstream file2("solution.txt");

	for(int i=0; i<nSpheres && i<spheres.size(); ++i)
	{
		file2 << "(";

		file2 << spheres[i].center.x << ", ";
		file2 << spheres[i].center.y << ", ";
		file2 << spheres[i].center.z << "): ";

		file2 << spheres[i].radius << "\n\n";
	}

	file2.close();
}

int main()
{
	vector3f points;

	LoadPlyFile(&points);
	Detect(points, 6);

	return 0;
}