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
	SUPEREXPORT void saveModel(double*** W, int* layers, int layer_count, char* fileName);
	SUPEREXPORT double*** loadModel(char* fileName);
}
