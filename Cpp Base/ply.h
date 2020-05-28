#pragma once

#include <iostream>
#include <fstream>
#include <vector>

namespace ply
{
	struct float3
	{
		float x;
		float y;
		float z;

		float3();

		float3(float x, float y, float z);

		bool operator == (float3 other);

		bool operator != (float3 other);
	};
	
	typedef std::vector<float3> vector3f;

	bool LoadVertex(vector3f*, std::string path);
}


