#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <cmath>	


extern "C" {
	SUPEREXPORT double predict_reg_RBF_naive(double* W, double* X, double* Xpredict, int inputCountPerSample, double gamma);
}