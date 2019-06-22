#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <iostream>
#include <fstream>


extern "C" {
	SUPEREXPORT void saveModel(void* model);
}
