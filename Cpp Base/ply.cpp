#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "ply.h"

namespace ply
{
	float3::float3()
	{
		this->x = 0.0f;
		this->y = 0.0f;
		this->z = 0.0f;
	}

	float3::float3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	bool float3::operator == (float3 other)
	{
		bool equals = true;

		equals = equals && this->x == other.x;
		equals = equals && this->y == other.y;
		equals = equals && this->z == other.z;

		return equals;
	}

	bool float3::operator != (float3 other)
	{
		bool equals = true;

		equals = equals && this->x == other.x;
		equals = equals && this->y == other.y;
		equals = equals && this->z == other.z;

		return !equals;
	}

	bool LoadVertex(vector3f* points, std::string path)
	{
		std::ifstream text(path, std::ios::binary);

		char header[2048];

		text.read(header, 2048);

		std::string str(header);
		
		size_t index1 = str.find("ply");
		size_t index2 = str.find("PLY");
		
		if(index1 < 0 && index2 < 0)
		{
			return false;
		}

		size_t index3 = str.find("vertex")+strlen("vertex");
		size_t index4 = str.find('\n', index3);
		size_t index5 = str.find("end_header\n")+strlen("end_header\n");

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
}

