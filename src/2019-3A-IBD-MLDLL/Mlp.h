#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <random>
#include <chrono>
#include <iostream>


extern "C" {
	SUPEREXPORT double*** create_mlp_model(int* layers, int layer_count, int inputCountPerSample);

	SUPEREXPORT void fit_mlp_classification(double*** W,
		double* Xtrain,
		int* YTrain,
		int* layers,
		int layer_count,
		int sampleCount,
		int inputCountPerSample,
		double alpha,
		int epochs);

	SUPEREXPORT void fit_mlp_regression(double*** W,
		double* Xtrain,
		int* YTrain,
		int* layers,
		int layer_count,
		int sampleCount,
		int inputCountPerSample,
		double alpha,
		int epochs);

	SUPEREXPORT double* predict_mlp_classification(double*** W, int* layers, int layer_count, int inputCountPerSample, double* Xinput);

	SUPEREXPORT double* predict_mlp_regression(double*** W, int* layers, int layer_count, int inputCountPerSample, double* Xinput);
}