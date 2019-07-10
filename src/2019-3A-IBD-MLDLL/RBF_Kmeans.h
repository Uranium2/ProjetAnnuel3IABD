#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <iostream>

extern "C" {
	SUPEREXPORT double* fit_regRBF_Kmeans(double* Kmeans, int K, double* X, double* YTrain, int sampleCount, int inputCountPerSample, double gamma);
}