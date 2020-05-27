#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "Reflection.h"

////////////////////////////////////////////////////////////////////////

template <typename Type>

Reflection<Type> Malloc(const unsigned int count)
{
	const unsigned int size = count * sizeof(Type);

	Reflection<Type> reflection;

	if(size == 0)
	{
		return reflection;
	}

	if(cudaMalloc(&reflection.device, size) != cudaSuccess)
	{
		reflection.device = nullptr;
		return reflection;
	}

	if(cudaMemset(reflection.device, 0, size) != cudaSuccess)
	{
		cudaFree(reflection.device);
		reflection.device = nullptr;
		return reflection;
	}

	reflection.host = new Type[count];

	memset(reflection.host, 0, size);

	reflection.size = size;

	return reflection;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

Reflection<Type> Malloc(Type* hostBuffer, const unsigned int count, bool send = false)
{
	const unsigned int size = count * sizeof(Type);

	Reflection<Type> reflection;

	if(size == 0)
	{
		return reflection;
	}

	if(cudaMalloc(&reflection.device, size) != cudaSuccess)
	{
		reflection.device = nullptr;
		return reflection;
	}

	if(!send)
	{
		if(cudaMemset(reflection.device, 0, size) != cudaSuccess)
		{
			cudaFree(reflection.device);
			reflection.device = nullptr;
			return reflection;
		}
	}

	reflection.host = new Type[count];

	memcpy(reflection.host, hostBuffer, size);

	reflection.size = size;

	if(send)
	{
		Send(reflection);
	}

	return reflection;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

void Free(Reflection<Type>& reflection)
{
	if(reflection.size)
	{
		if(reflection.host != nullptr)
		{
			delete []reflection.host;
			reflection.host = nullptr;
		}

		if(reflection.device != nullptr)
		{
			cudaFree(reflection.device);
			reflection.device = nullptr;
		}

		reflection.size = 0;
	}
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

bool IsValid(Reflection<Type>& reflection)
{
	if(reflection.size == 0 || reflection.host == nullptr || reflection.device == nullptr)
	{
		return false;
	}

	return true;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

bool Send(Reflection<Type>& reflection)
{
	if(!IsValid(reflection))
	{
		return false;
	}

	return cudaMemcpy(reflection.device, reflection.host, reflection.size, cudaMemcpyHostToDevice) == cudaSuccess;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

bool Send(Reflection<Type>& reflection, const unsigned int count)
{
	if(!IsValid(reflection))
	{
		return false;
	}

	if(count == 0)
	{
		return true;
	}

	unsigned int size = count * sizeof(Type);

	if(size > reflection.size)
	{
		size = reflection.size;

		throw "Invalid Argument Exeption";
	}

	return cudaMemcpy(reflection.device, reflection.host, size, cudaMemcpyHostToDevice) == cudaSuccess;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

bool Receive(Reflection<Type>& reflection)
{
	if(!IsValid(reflection))
	{
		return false;
	}

	return cudaMemcpy(reflection.host, reflection.device, reflection.size, cudaMemcpyDeviceToHost) == cudaSuccess;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

bool Receive(Reflection<Type>& reflection, const unsigned int count)
{
	if(!IsValid(reflection))
	{
		return false;
	}

	if(count == 0)
	{
		return true;
	}

	unsigned int size = count * sizeof(Type);

	if(size > reflection.size)
	{
		size = reflection.size;

		throw "Invalid Argument Exeption";
	}

	return cudaMemcpy(reflection.host, reflection.device, size, cudaMemcpyDeviceToHost) == cudaSuccess;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

Type* Host(Reflection<Type>& reflection)
{
	return reflection.host;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

Type* Device(Reflection<Type>& reflection)
{
	return reflection.device;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

void Show(Reflection<Type>& reflection, unsigned int count = 0)
{
	if(!IsValid(reflection))
	{
		std::cout << "Invalid Instance\n\n";
		return;
	}

	const unsigned int max_count = reflection.size / sizeof(Type);

	if(count > max_count || count == 0)
	{
		count = max_count;
	}

	for(int i=0; i<count && i<1024; ++i)
	{
		std::cout << "[" << i << "]: " << reflection.host[i] << "\n";
	}
}

////////////////////////////////////////////////////////////////////////

