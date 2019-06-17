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
	SUPEREXPORT int predict_class_RBF_naive(double* W, double* X, double* Xpredict, int inputCountPerSample, double gamma, int N);
	SUPEREXPORT double predict_reg_RBF_naive(double* W, double* X, double* Xpredict, int inputCountPerSample, double gamma, int N);
	SUPEREXPORT double* fit_reg_RBF_naive(double* XTrain, double gamma, double* YTrain, int sampleCount, int inputCountPerSample);
}