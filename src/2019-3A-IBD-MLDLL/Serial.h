#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


extern "C" {
	SUPEREXPORT void saveLinearModel(double* W, int inputCountPerSample, char* fileName);
	SUPEREXPORT int getInputCountPerSample(char* fileName);
	SUPEREXPORT double* loadLinearModel(char* fileName);
	SUPEREXPORT void saveModel(double*** W, int* layers, int layer_count, char* fileName);
	SUPEREXPORT int* getLayers(char* fileName);
	SUPEREXPORT int getLayer_count(char* fileName);
	SUPEREXPORT double*** loadModel(char* fileName);
}
