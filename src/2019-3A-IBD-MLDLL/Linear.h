#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>    

extern "C" {
	SUPEREXPORT double* create_linear_model(int inputCountPerSample);

	SUPEREXPORT void fit_classification_rosenblatt_rule(
		double* W,
		double* XTrain,
		int sampleCount,
		int inputCountPerSample,
		double* YTrain,
		double alpha,
		int epochs);

	SUPEREXPORT double predict_classification_rosenblatt(double* W, double* X, int inputCountPerSample);

	SUPEREXPORT double* fit_regression(
		double* XTrain,
		int sampleCount,
		int inputCountPerSample,
		double* YTrain
	);

	SUPEREXPORT double predict_regression(double* W, double* X, int inputCountPerSample);
}