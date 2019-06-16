#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <iostream>


extern "C" {
	SUPEREXPORT double* get_Kmeans(int K, double* Xtrain, int sampleCount, int inputCountperSample);
}